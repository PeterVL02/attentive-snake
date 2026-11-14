"""Lunar Lander: the env contract, its rewards and outcomes, lazy rendering, and watching every kind on it."""

import dataclasses
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.envs.box2d.lunar_lander import LunarLander

from attentive_snake.config import EnvConfig, ExperimentConfig, save_experiment
from attentive_snake.config.schema import experiment_schema
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.games.lunar_lander.config import LanderRules
from attentive_snake.games.lunar_lander.game import GAME, OBSERVATION_LABELS
from attentive_snake.games.lunar_lander.outcome import step_event
from attentive_snake.games.lunar_lander.reward import LanderReward
from attentive_snake.games.registry import get_game
from attentive_snake.games.validation import spec_problems
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, TrainConfig, Trainer
from attentive_snake.views.image import png_bytes
from attentive_snake.watch import WatchSession

TERMINAL = {"landed", "crashed", "flew_off", "timeout"}


def make(rules=LanderRules(), reward=LanderReward(), seed=0):
    return GAME.make_env(rules, reward, seed)


def play(env, actions, seed=0):
    """Reset with ``seed`` and play ``actions`` (cycled) until the episode ends; returns the steps."""
    env.reset(seed)
    steps = []
    while not steps or not (steps[-1].terminated or steps[-1].truncated):
        steps.append(env.step(actions[len(steps) % len(actions)]))
    return steps


def test_registered_as_a_vector_game_with_an_image_view():
    assert get_game("lunar_lander") is GAME
    assert (GAME.observation_kind, GAME.view_kind) == ("vector", "image")
    assert len(GAME.observation_labels) == 8 and spec_problems(GAME, "lunar_lander") == []


def test_shapes_and_dtype():
    env = make()
    assert (env.observation_shape, env.num_actions) == ((8,), 4)
    first = env.reset(1)
    step = env.step(2)
    for observation in (first, step.observation):
        assert observation.shape == (8,) and observation.dtype == np.float32


def test_same_seed_same_episode_and_other_seed_differs():
    a, b = play(make(), [0, 2, 1, 3], seed=7), play(make(), [0, 2, 1, 3], seed=7)
    assert [(s.reward, s.event) for s in a] == [(s.reward, s.event) for s in b]
    assert np.array_equal(a[-1].observation, b[-1].observation)
    c = play(make(), [0, 2, 1, 3], seed=8)
    assert not np.array_equal(a[5].observation, c[5].observation)


def test_constructor_seed_is_used_by_the_first_plain_reset():
    assert np.array_equal(make(seed=3).reset(), make(seed=3).reset())


def test_score_is_the_raw_return_and_reward_is_scaled():
    steps = play(make(reward=LanderReward(scale=0.5)), [0], seed=0)  # doing nothing crashes
    assert steps[-1].event in TERMINAL - {"timeout"} and steps[-1].terminated
    assert all(s.event == "flying" for s in steps[:-1])
    assert steps[-1].score == pytest.approx(sum(s.reward for s in steps) / 0.5)


def test_timeout_truncates_and_only_changes_the_reward():
    steps = play(make(LanderRules(max_episode_steps=5), LanderReward(scale=1.0, timeout=-10.0)), [0], seed=0)
    last = steps[-1]
    assert len(steps) == 5 and last.truncated and not last.terminated and last.event == "timeout"
    assert last.score == pytest.approx(sum(s.reward for s in steps) + 10.0)


@pytest.mark.parametrize(
    "awake, x, terminated, truncated, event",
    [
        (True, 0.0, False, False, "flying"),
        (True, 0.0, False, True, "timeout"),
        (False, 0.0, True, False, "landed"),
        (False, 1.2, True, True, "landed"),  # at rest wins, as in Gymnasium's reward
        (True, 1.0, True, False, "flew_off"),
        (True, 0.3, True, False, "crashed"),
    ],
)
def test_outcome_words(awake, x, terminated, truncated, event):
    lander = SimpleNamespace(lander=SimpleNamespace(awake=awake))
    assert step_event(lander, x, terminated, truncated) == event


def test_rules_and_reward_validation():
    for bad in ({"gravity": 0.0}, {"gravity": -12.0}, {"wind_power": 21.0}, {"turbulence_power": -1.0}, {"max_episode_steps": 0}):
        with pytest.raises(ValueError):
            LanderRules(**bad)
    with pytest.raises(ValueError):
        LanderReward(scale=0.0)


def test_view_is_a_png_of_the_screen_size():
    env = make()
    env.reset(0)
    view = env.view()
    data = png_bytes(view)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    assert struct.unpack(">II", data[16:24]) == (view["width"], view["height"]) == (600, 400)


def test_training_envs_never_render(monkeypatch):
    def no_render(self):
        raise AssertionError("rendered during training")

    monkeypatch.setattr(LunarLander, "render", no_render)
    config = ExperimentConfig(game_name="lunar_lander", game=LanderRules(max_episode_steps=30), reward=LanderReward(), env=EnvConfig(num_envs=3))
    env = make_vec_env(config)
    env.reset(0)
    for _ in range(70):  # episodes end and auto-reset on the way
        env.step(np.random.randint(0, 4, size=3))
    assert all(single.game._env.unwrapped.render_mode is None for single in env.envs)


def test_labels_are_checked():
    assert "only for vector games" in " ".join(spec_problems(dataclasses.replace(GAME, observation_kind="grid"), "lunar_lander"))
    assert "8" in " ".join(spec_problems(dataclasses.replace(GAME, observation_labels=OBSERVATION_LABELS[:3]), "lunar_lander"))
    assert "distinct" in " ".join(spec_problems(dataclasses.replace(GAME, observation_labels=("x",) * 8), "lunar_lander"))


def test_schema_recommends_the_mlp_kinds_and_describes_vector_reading():
    kinds = experiment_schema("lunar_lander")["model_kinds"]
    assert {kind for kind, row in kinds.items() if row["recommended"]} == {"mlp"}
    for kind, row in kinds.items():
        assert row["description"] == KIND_INFO[kind].inputs["vector"].description
        assert "vector" in row["observation_kinds"]
    assert "kernel_size" not in kinds["cnn"]["fields"] and "kernel_size" in experiment_schema("snake")["model_kinds"]["cnn"]["fields"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A tiny trained Lunar Lander run per kind (temporal kinds with 3 frames and actions)."""
    root = tmp_path_factory.mktemp("lander_runs")
    for kind in POLICY_KINDS:
        history = 3 if KIND_INFO[kind].uses_history else 1
        config = ExperimentConfig(
            game_name="lunar_lander",
            game=LanderRules(),
            reward=LanderReward(),
            model=ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=1, conv_channels=8, conv_layers=1),
            train=TrainConfig(total_env_steps=32, rollout_length=8, num_minibatches=2, checkpoint_dir=root / kind, device="cpu"),
            env=EnvConfig(num_envs=2, history_length=history, history_actions=history > 1),
        )
        save_experiment(config, config_path(root / kind))
        env = make_vec_env(config)
        Trainer(make_policy(config, env), env, config.train, NullLogger()).train()
    return root


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_watching_every_kind_gives_the_declared_attention(runs, kind):
    session = WatchSession(runs / kind, seed=0)
    assert json.loads(json.dumps(session.info)) == session.info
    assert (session.info["view_kind"], session.info["width"], session.info["height"]) == ("image", 600, 400)
    assert session.info["element_names"] == list(OBSERVATION_LABELS)
    view = KIND_INFO[kind].attention_view
    for frame in [session.reset(), *(session.step() for _ in range(3))]:
        data = frame.to_dict()
        assert json.loads(json.dumps(data)) == data and data["png"] and data["heatmap"] is None
        if view in ("grid", "frames"):
            weights = data["element_weights"]
            assert len(weights) == 8 and min(weights) >= 0.0 and max(weights) == pytest.approx(1.0)
            assert data["frame_weights"] is None
        elif view == "weights":
            assert len(data["frame_weights"]) == 3 and sum(data["frame_weights"]) == pytest.approx(1.0, abs=1e-5)
            assert data["element_weights"] is None
        else:
            assert data["element_weights"] is None and data["frame_weights"] is None


def test_temporal_element_weights_follow_the_selected_frame(runs):
    session = WatchSession(runs / "temporal_transformer", seed=0)
    session.step()
    newest = session.current().element_weights
    session.set_options(frame=0)
    oldest = session.current().element_weights
    assert len(oldest) == 8 and oldest != newest
