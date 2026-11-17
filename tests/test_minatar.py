"""MinAtar: the adapter (titles, the trails rule, actions, seeding, rewards, outcomes, the board view),
its fork locks, its schema, and training and watching every grid kind on it."""

import dataclasses
import json

import numpy as np
import pytest

from attentive_snake.config import EnvConfig, ExperimentConfig, save_experiment
from attentive_snake.config.fork_rules import locked_field_changes, locked_fields, require_forkable
from attentive_snake.config.schema import experiment_schema
from attentive_snake.factory import make_env, make_policy, make_trainer, make_vec_env
from attentive_snake.games.minatar.actions import FULL_ACTION_LABELS
from attentive_snake.games.minatar.channels import visible_channels
from attentive_snake.games.minatar.config import MinAtarRules
from attentive_snake.games.minatar.game import GAME
from attentive_snake.games.minatar.outcome import step_event
from attentive_snake.games.minatar.reward import MinAtarReward, minatar_reward
from attentive_snake.games.minatar.reward_table import reward_table
from attentive_snake.games.minatar.titles import TITLE_KEYS, TITLES, Title, get_title
from attentive_snake.games.registry import get_game
from attentive_snake.games.validation import spec_problems
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, TrainConfig
from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.watch import WatchSession


def make(rules=MinAtarRules(), reward=MinAtarReward(), seed=0):
    return GAME.make_env(rules, reward, seed)


def play(env, steps, seed=None, action=0):
    env.reset(seed)
    return [env.step(action) for _ in range(steps)]


def package_channels(title: str) -> dict[str, int]:
    from importlib import import_module

    return import_module(f"minatar.environments.{title}").Env().channels


# --------------------------------------------------------------------------- the game, titles and channels


def test_registered_as_a_grid_game_with_a_board_view():
    assert get_game("minatar") is GAME
    assert (GAME.observation_kind, GAME.view_kind) == ("grid", "board")
    assert spec_problems(GAME, "minatar") == []


@pytest.mark.parametrize("title", TITLES, ids=TITLE_KEYS)
def test_every_motion_channel_is_a_channel_of_the_installed_package(title):
    channels = package_channels(title.key)
    assert set(title.motion_channels) <= set(channels)
    assert visible_channels(channels, title, trails=True) == tuple(sorted(channels, key=channels.get))


def test_the_motion_channels_are_the_ones_the_package_names_for_motion():
    """Read from the package's channel names: trails, Freeway's speed trails, Space Invaders' march direction."""
    for title in TITLES:
        motion = {name for name in package_channels(title.key) if name == "trail" or name.startswith(("speed", "alien_"))}
        assert motion == set(title.motion_channels), title.key


def test_an_unknown_motion_channel_is_reported():
    with pytest.raises(ValueError, match="has no channel"):
        visible_channels({"ball": 0}, Title("x", "X", ("trail",)), trails=False)


@pytest.mark.parametrize("title", TITLES, ids=TITLE_KEYS)
@pytest.mark.parametrize("trails", [True, False])
def test_trails_off_removes_the_motion_channels_from_the_observation(title, trails):
    env = make(MinAtarRules(title=title.key, trails=trails))
    channels = package_channels(title.key)
    expected = len(channels) - (0 if trails else len(title.motion_channels))
    assert env.observation_shape == (expected, 10, 10) and len(env.channel_names) == expected
    assert trails or not set(title.motion_channels) & set(env.channel_names)
    observation = env.reset(0)
    assert observation.shape == env.observation_shape and observation.dtype == np.float32
    assert set(np.unique(observation)) <= {0.0, 1.0}


def test_trails_off_shows_the_same_cells_minus_the_trail():
    with_trails, without = make(MinAtarRules(trails=True)), make(MinAtarRules(trails=False))
    for a, b in zip(play(with_trails, 20, seed=3), play(without, 20, seed=3)):
        assert with_trails.channel_names == ("paddle", "ball", "trail", "brick")
        assert np.array_equal(np.delete(a.observation, 2, axis=0), b.observation) and a.reward == b.reward


# --------------------------------------------------------------------------- actions, seeding, rewards, outcomes


def test_minimal_and_full_action_sets_and_their_names():
    assert make().action_labels == ("nothing", "left", "right") == GAME.action_labels
    assert make(MinAtarRules(title="freeway")).action_labels == ("nothing", "up", "down")
    full = make(MinAtarRules(title="freeway", full_action_space=True))
    assert full.num_actions == 6 and full.action_labels == FULL_ACTION_LABELS


def test_same_seed_same_game_and_the_constructor_seed_is_used_once():
    def trace(env, seed=None):
        rng = np.random.default_rng(1)
        env.reset(seed)
        return [(env.step(int(rng.integers(3))).observation.tobytes()) for _ in range(60)]

    assert trace(make(seed=None), 5) == trace(make(seed=None), 5)
    assert trace(make(seed=5)) == trace(make(seed=None), 5)
    env = make()
    first = trace(env, 7)
    trace(env, 8)
    assert trace(env, 7) == first  # the last action of the previous game does not stick into the next


def test_breakout_ends_when_the_ball_is_lost_and_the_step_limit_truncates():
    steps = play(make(MinAtarRules(sticky_action_prob=0.0)), 200, seed=0)
    last = next(step for step in steps if step.terminated)
    assert last.event == "game_over" and not last.truncated
    cut = play(make(MinAtarRules(max_episode_steps=3)), 3, seed=0)[-1]
    assert (cut.truncated, cut.terminated, cut.event) == (True, False, "timeout")


def test_reward_scales_points_and_pays_game_over_unscaled():
    assert minatar_reward(MinAtarReward(scale=0.5, game_over=-1.0), 2, False) == 1.0
    assert minatar_reward(MinAtarReward(scale=0.5, game_over=-1.0), 0, True) == -1.0
    with pytest.raises(ValueError):
        MinAtarReward(scale=0.0)
    rows = {row["situation"]: row for row in reward_table(MinAtarReward(scale=2.0, game_over=-3.0), 1)}
    assert rows["Score 1 point"]["reward"] == 2.0 and rows["Lose the game (ball lost, player hit)"]["reward"] == -3.0


def test_score_is_the_raw_points_whatever_the_reward():
    rng = np.random.default_rng(0)
    env = make(MinAtarRules(title="freeway", sticky_action_prob=0.0), MinAtarReward(scale=0.5))
    env.reset(0)
    steps = [env.step(1 if rng.random() < 0.9 else 0) for _ in range(300)]  # mostly up
    assert steps[-1].score == sum(step.reward for step in steps) * 2 > 0
    assert {step.event for step in steps} >= {"scored", "playing"}


@pytest.mark.parametrize(
    ("points", "game_over", "timed_out", "event"),
    [(0, False, False, "playing"), (1, False, False, "scored"), (1, True, False, "game_over"), (0, False, True, "timeout")],
)
def test_outcome_words(points, game_over, timed_out, event):
    assert step_event(points, game_over, timed_out) == event


def test_rules_validation():
    for bad in ({"title": "pong"}, {"sticky_action_prob": 1.0}, {"max_episode_steps": 0}):
        with pytest.raises(ValueError):
            MinAtarRules(**bad)
    assert get_title("seaquest").label == "Seaquest"


# --------------------------------------------------------------------------- the board view


def test_view_is_a_board_with_one_named_coloured_layer_per_visible_channel():
    env = make(MinAtarRules(trails=False))
    observation = env.reset(0)
    view = env.view()
    assert (view["width"], view["height"], view["snake"], view["foods"]) == (10, 10, [], [])
    assert [layer["name"] for layer in view["layers"]] == ["paddle", "ball", "brick"]
    for plane, layer in zip(observation, view["layers"]):
        assert sorted(map(tuple, layer["cells"])) == sorted((int(x), int(y)) for y, x in np.argwhere(plane > 0))
    with_trails = make(MinAtarRules(trails=True))
    with_trails.reset(0)
    colors = {layer["name"]: layer["color"] for layer in with_trails.view()["layers"]}
    assert all(colors[layer["name"]] == layer["color"] for layer in view["layers"])  # a channel keeps its colour


def test_the_board_window_draws_layers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from attentive_snake.views.board_window import BoardWindow

    env = make()
    env.reset(0)
    window = BoardWindow(10, 10, cell_size=10)
    window.render(env.view())
    paddle = next(layer for layer in env.view()["layers"] if layer["name"] == "paddle")
    x, y = paddle["cells"][0]
    assert pygame.display.get_surface().get_at((x * 10 + 5, y * 10 + 5))[:3] == pygame.Color(paddle["color"])[:3]
    window.close()


def test_finding_the_game_does_not_import_the_package():
    import subprocess
    import sys

    code = "import sys; import attentive_snake.games.minatar; sys.exit(int('minatar' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --------------------------------------------------------------------------- fork locks, schema


def minatar_config(**rules) -> ExperimentConfig:
    return ExperimentConfig(game_name="minatar", game=MinAtarRules(**rules), reward=MinAtarReward(), model=ModelConfig(kind="cnn"))


def test_a_fork_must_keep_the_title_the_trails_and_the_action_space():
    locked = locked_fields("minatar")
    assert {"game.title", "game.trails", "game.full_action_space", "env.history_length"} <= set(locked)
    assert not {"game.sticky_action_prob", "game.difficulty_ramping", "game.max_episode_steps"} & set(locked)
    parent = minatar_config()
    for change, key in [({"title": "asterix"}, "game.title"), ({"trails": False}, "game.trails"), ({"full_action_space": True}, "game.full_action_space")]:
        assert locked_field_changes(parent, minatar_config(**change)) == [key]
        with pytest.raises(ValueError, match=key):
            require_forkable(parent, minatar_config(**change))
    require_forkable(parent, minatar_config(sticky_action_prob=0.0, difficulty_ramping=False, max_episode_steps=100))
    require_forkable(parent, dataclasses.replace(parent, reward=MinAtarReward(scale=0.5, game_over=-1.0)))


def test_schema_offers_the_titles_the_trails_rule_and_every_kind():
    schema = experiment_schema("minatar")
    rules = {row["name"]: row for row in schema["sections"]["game"]}
    assert rules["title"]["choices"] == list(TITLE_KEYS) and rules["title"]["default"] == "breakout"
    assert rules["trails"]["default"] is True and "history_length" in rules["trails"]["description"]
    assert rules["sticky_action_prob"]["default"] == 0.1 and rules["difficulty_ramping"]["default"] is True
    assert [row["name"] for row in schema["sections"]["reward"]] == ["scale", "game_over"]
    assert set(schema["model_kinds"]) == set(POLICY_KINDS)  # every kind reads grids


# --------------------------------------------------------------------------- training and watching every grid kind

SMALL = MinAtarRules(trails=False, max_episode_steps=6)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A tiny trained Breakout run (trails off, 2 frames) per kind."""
    root = tmp_path_factory.mktemp("minatar_runs")
    for kind in POLICY_KINDS:
        config = ExperimentConfig(
            game_name="minatar",
            game=SMALL,
            reward=MinAtarReward(),
            model=ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=1, conv_channels=8, conv_layers=1),
            train=TrainConfig(total_env_steps=32, checkpoint_dir=root / kind, device="cpu"),
            algorithm_settings=PpoConfig(rollout_length=8, num_minibatches=2),
            env=EnvConfig(num_envs=2, history_length=2),
        )
        save_experiment(config, config_path(root / kind))
        env = make_vec_env(config)
        make_trainer(config, make_policy(config, env), env, NullLogger()).train()
    return root


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_every_kind_trains_and_is_watched_with_its_declared_attention(runs, kind):
    session = WatchSession(runs / kind, seed=0)
    assert json.loads(json.dumps(session.info)) == session.info
    assert (session.info["view_kind"], session.info["width"], session.info["height"]) == ("board", 10, 10)
    assert session.info["observation_shape"] == [6, 10, 10]  # 2 frames of paddle, ball, brick
    assert session.info["action_names"] == ["nothing", "left", "right"]
    view = KIND_INFO[kind].attention_view
    frames = [session.reset(), *(session.step() for _ in range(7))]
    assert any(frame.event in ("timeout", "game_over") for frame in frames)  # 6 steps at most
    for frame in frames:
        data = frame.to_dict()
        assert json.loads(json.dumps(data)) == data and [layer["name"] for layer in data["layers"]] == ["paddle", "ball", "brick"]
        if frame.done:
            continue
        if view in ("grid", "frames"):  # "frames": cells of every frame, one frame's heatmap shown
            assert np.array(data["heatmap"]).shape == (10, 10)  # one weight per cell, as on a Snake board
        elif view == "weights":
            assert len(data["frame_weights"]) == 2
        else:
            assert data["heatmap"] is None and data["frame_weights"] is None


def test_the_observation_stacks_frames_oldest_first():
    config = minatar_config(trails=False)
    config = dataclasses.replace(config, env=EnvConfig(history_length=3))
    env = make_env(config, seed=0)
    first = env.reset()
    assert first.shape == (9, 10, 10)
    second = env.step(0).observation
    assert np.array_equal(second[3:6], first[6:9])  # the newest frame moved one place back
