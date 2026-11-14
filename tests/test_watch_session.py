import json
import sys

import pytest

from attentive_snake.config import EnvConfig, ExperimentConfig, save_experiment
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.games.snake.config import GameConfig
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, TrainConfig, Trainer
from attentive_snake.watch import WatchFrame, WatchSession


def make_run(run_dir, kind: str, history: int = 1, actions: bool = False) -> ExperimentConfig:
    """Train 2 tiny updates through the library and save config + checkpoint in ``run_dir``."""
    config = ExperimentConfig(
        game=GameConfig(width=5, height=5, max_steps_without_food=30),
        model=ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2),
        train=TrainConfig(total_env_steps=32, rollout_length=8, num_minibatches=2, checkpoint_dir=run_dir, device="cpu"),
        env=EnvConfig(num_envs=2, history_length=history, history_actions=actions),
    )
    save_experiment(config, config_path(run_dir))
    env = make_vec_env(config)
    Trainer(make_policy(config, env), env, config.train, NullLogger()).train()
    return config


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    specs = {kind: (1, False) if not KIND_INFO[kind].uses_history else (3, True) for kind in POLICY_KINDS}
    for kind, (history, actions) in specs.items():
        make_run(root / kind, kind, history, actions)
    return root


def run_frames(session: WatchSession, count: int) -> list[WatchFrame]:
    return [session.step() for _ in range(count)]


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_every_kind_yields_json_frames_with_the_declared_attention(runs, kind) -> None:
    session = WatchSession(runs / kind, seed=0)
    assert json.loads(json.dumps(session.info)) == session.info
    assert session.info["kind"] == kind
    assert session.info["attention_view"] == KIND_INFO[kind].attention_view
    assert (session.info["width"], session.info["height"]) == (5, 5)
    assert session.info["env_steps"] == 32
    frames = [session.reset(), *run_frames(session, 40)]
    for frame in frames:
        assert json.loads(json.dumps(frame.to_dict())) == frame.to_dict()
        live = not frame.done
        view = KIND_INFO[kind].attention_view
        if view in ("grid", "frames"):
            assert (frame.heatmap is not None) == live
            if live:
                assert len(frame.heatmap) == 5 and len(frame.heatmap[0]) == 5
                assert 0.0 <= min(map(min, frame.heatmap)) and max(map(max, frame.heatmap)) <= 1.0
            assert frame.frame_weights is None
        elif view == "weights":
            assert (frame.frame_weights is not None) == live
            if live:
                assert len(frame.frame_weights) == 3 and sum(frame.frame_weights) == pytest.approx(1.0, abs=1e-5)
            assert frame.heatmap is None
        else:
            assert frame.heatmap is None and frame.frame_weights is None
            assert frame.caption == ("model has no attention" if live else "")


def test_episode_flow_reset_done_and_auto_restart(runs) -> None:
    session = WatchSession(runs / "transformer", seed=1)
    first = session.reset()
    assert first.event is None and not first.done and first.steps == 0 and first.episode == 0
    assert first.snake[0] == list(session.state.snake[0]) and len(first.snake) == 3
    frames = run_frames(session, 200)
    done_indices = [i for i, f in enumerate(frames) if f.done]
    assert done_indices, "the 30-step starvation limit guarantees episodes end"
    i = done_indices[0]
    assert frames[i].event in {"hit_wall", "hit_self", "starved", "won"}
    assert frames[i].episode == 1
    assert frames[i + 1].steps == 1 and not frames[i + 1].done and frames[i + 1].episode == 1  # new episode started


def test_info_for_temporal_and_options_validation(runs) -> None:
    session = WatchSession(runs / "temporal_transformer", seed=0)
    assert session.info["history_length"] == 3 and session.info["num_layers"] == 2
    session.set_options(layer=0, frame=0, sample=True, attention=True)
    assert session.step().caption == "layer 0, frame 0"
    for bad in ({"layer": 2}, {"layer": -3}, {"frame": 3}, {"frame": -4}):
        with pytest.raises(ValueError):
            session.set_options(**bad)
    assert session.step().caption == "layer 0, frame 0"  # failed set_options changed nothing
    with pytest.raises(ValueError):
        WatchSession(runs / "temporal_transformer", layer=5)


def test_attention_can_be_switched_off_and_cnn_has_no_layers(runs) -> None:
    session = WatchSession(runs / "transformer", attention=False, seed=0)
    assert session.reset().heatmap is None
    session.set_options(attention=True)
    assert session.current().heatmap is not None
    cnn = WatchSession(runs / "cnn", seed=0)
    assert cnn.info["num_layers"] == 0 and cnn.info["attention_view"] is None
    cnn.set_options(layer=3, frame=3)  # nothing to index, nothing to reject


def test_explicit_checkpoint_missing_checkpoint_and_seed_determinism(runs, tmp_path) -> None:
    checkpoint = sorted((runs / "cnn").glob("checkpoint_*.pt"))[0]
    session = WatchSession(runs / "cnn", checkpoint=checkpoint, seed=3)
    assert session.info["checkpoint"] == str(checkpoint)
    a = [f.snake for f in run_frames(WatchSession(runs / "cnn", seed=7), 15)]
    b = [f.snake for f in run_frames(WatchSession(runs / "cnn", seed=7), 15)]
    assert a == b
    empty = tmp_path / "empty"
    save_experiment(ExperimentConfig(), config_path(empty))
    with pytest.raises(FileNotFoundError):
        WatchSession(empty)


def test_session_does_not_import_pygame() -> None:
    import subprocess

    code = "import sys, attentive_snake.watch; sys.exit('pygame' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
