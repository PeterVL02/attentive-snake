"""Atari: the adapter (titles, preprocessing, frame skip, seeding, actions, rewards, outcomes), its fork locks,
its schema, and watching image models on it."""

import dataclasses
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from attentive_snake.ale.actions import ALE_ACTION_NAMES, FULL_ACTION_LABELS, action_labels
from attentive_snake.ale.screen import ScreenShrinker, area_weights
from attentive_snake.config import EnvConfig, ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.fork_rules import locked_field_changes, locked_fields, require_forkable
from attentive_snake.config.overrides import apply_overrides, parse_override
from attentive_snake.config.schema import experiment_schema
from attentive_snake.episode_metrics import EpisodeRecord
from attentive_snake.factory import make_env, make_policy, make_trainer, make_vec_env
from attentive_snake.games.atari import env as atari_env
from attentive_snake.games.atari.config import AtariRules
from attentive_snake.games.atari.emulator import make_emulator
from attentive_snake.games.atari.game import GAME
from attentive_snake.games.atari.metrics import atari_statistics
from attentive_snake.games.atari.outcome import step_event
from attentive_snake.games.atari.reward import AtariReward, atari_reward
from attentive_snake.games.atari.titles import TITLE_KEYS, TITLES, get_title
from attentive_snake.games.atari.variants import available_variants
from attentive_snake.games.registry import get_game
from attentive_snake.games.validation import spec_problems
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, TrainConfig
from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.views.image import png_bytes
from attentive_snake.watch import WatchSession

REPEATABLE = AtariRules(sticky_actions=0.0)
IMAGE_KINDS = [kind for kind in POLICY_KINDS if "image" in KIND_INFO[kind].inputs]


def make(rules=REPEATABLE, reward=AtariReward(), seed=0):
    return GAME.make_env(rules, reward, seed)


def play(env, actions, steps, seed=None):
    env.reset(seed)
    return [env.step(actions[i % len(actions)]) for i in range(steps)]


# --------------------------------------------------------------------------- the game and its titles


def test_registered_as_an_image_game_with_an_image_view():
    assert get_game("atari") is GAME
    assert (GAME.observation_kind, GAME.view_kind) == ("image", "image")
    assert spec_problems(GAME, "atari") == []


def test_the_titles_the_owner_named_exist_in_the_emulator():
    import ale_py
    import gymnasium

    gymnasium.register_envs(ale_py)
    assert {"pong", "ms_pacman", "montezuma_revenge", "pitfall", "private_eye", "venture"} <= set(TITLE_KEYS)
    for title in TITLES:
        assert title.env_id in gymnasium.registry and title.label
    assert get_title("pong").env_id == "ALE/Pong-v5"
    with pytest.raises(ValueError, match="unknown Atari title 'chess'; known titles: pong"):
        AtariRules(title="chess")


@pytest.mark.parametrize("title", TITLE_KEYS)
def test_every_title_plays_with_the_same_18_actions_and_picture(title):
    env = make(AtariRules(title=title, max_episode_steps=3))
    assert (env.observation_shape, env.num_actions) == ((1, 84, 84), 18)
    assert env.action_labels == FULL_ACTION_LABELS
    steps = play(env, [0, 1, 17], 3)
    assert steps[-1].truncated and steps[-1].event == "timeout" and steps[-1].observation.shape == (1, 84, 84)


def test_the_titles_added_for_task_sequences_are_selectable():
    added = {
        "breakout": ("ALE/Breakout-v5", "Breakout"),
        "space_invaders": ("ALE/SpaceInvaders-v5", "Space Invaders"),
        "freeway": ("ALE/Freeway-v5", "Freeway"),
        "qbert": ("ALE/Qbert-v5", "Q*bert"),
        "asterix": ("ALE/Asterix-v5", "Asterix"),
        "beam_rider": ("ALE/BeamRider-v5", "Beam Rider"),
        "alien": ("ALE/Alien-v5", "Alien"),
        "bank_heist": ("ALE/BankHeist-v5", "Bank Heist"),
        "seaquest": ("ALE/Seaquest-v5", "Seaquest"),
    }
    assert {key: (get_title(key).env_id, get_title(key).label) for key in added} == added
    assert len(set(TITLE_KEYS)) == len(TITLES) == 15


# --------------------------------------------------------------------------- game mode and difficulty


def emulator_arguments(monkeypatch, rules):
    """What ``make_emulator`` passes to ``gymnasium.make`` for ``rules`` (the emulator itself is not made)."""
    import gymnasium

    calls = []
    monkeypatch.setattr(gymnasium, "make", lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(unwrapped=None))
    make_emulator(rules)
    return calls[0]


def test_without_a_mode_or_difficulty_the_emulator_is_made_exactly_as_before(monkeypatch):
    assert emulator_arguments(monkeypatch, AtariRules()) == (
        ("ALE/Pong-v5",),
        {"frameskip": 1, "repeat_action_probability": 0.25, "full_action_space": True, "max_num_frames_per_episode": 0, "disable_env_checker": True},
    )


@pytest.mark.parametrize("chosen", [{"mode": 1}, {"difficulty": 3}, {"mode": 1, "difficulty": 0}])
def test_a_chosen_mode_or_difficulty_is_passed_on_and_the_other_left_out(monkeypatch, chosen):
    _, arguments = emulator_arguments(monkeypatch, AtariRules(**chosen))
    assert {key: arguments[key] for key in ("mode", "difficulty") if key in arguments} == chosen


@pytest.mark.parametrize("title, chosen", [("freeway", {"mode": 3}), ("pong", {"difficulty": 3})])
def test_mode_and_difficulty_change_the_game_and_survive_a_reset(title, chosen):
    def screens(rules):
        env = make(rules)
        return [np.stack([step.observation for step in play(env, [1], 60, seed=5)]) for _ in range(2)]

    default = screens(AtariRules(title=title, sticky_actions=0.0))
    variant = screens(AtariRules(title=title, sticky_actions=0.0, **chosen))
    assert np.array_equal(variant[0], variant[1])  # the second game is the same variant
    assert not np.array_equal(default[1], variant[1])


def test_a_mode_keeps_the_picture_and_the_actions_of_the_title():
    default, variant = make(AtariRules(title="freeway")), make(AtariRules(title="freeway", mode=3, difficulty=1))
    assert (variant.observation_shape, variant.num_actions) == (default.observation_shape, default.num_actions)


@pytest.mark.parametrize(
    "chosen, message",
    [
        ({"mode": 9}, r"Atari game 'pong' has no mode 9; available modes: 0, 1\."),
        ({"difficulty": 7}, r"Atari game 'pong' has no difficulty 7; available difficulties: 0, 1, 2, 3\."),
        ({"mode": 9, "difficulty": 7}, r"no mode 9; available modes: 0, 1\. Atari game 'pong' has no difficulty 7"),
    ],
)
def test_an_unavailable_mode_or_difficulty_is_refused_with_the_available_ones(chosen, message):
    with pytest.raises(ValueError, match=message):
        AtariRules(**chosen)


@pytest.mark.parametrize("title", TITLE_KEYS)
def test_the_rom_of_every_title_answers_which_variants_it_has(title):
    variants = available_variants(title)
    assert variants.modes and variants.difficulties


def test_rules_without_a_mode_or_difficulty_do_not_load_the_emulator():
    script = "import sys; from attentive_snake.games.atari.config import AtariRules; AtariRules(title='qbert'); sys.exit('ale_py' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 0


def test_mode_and_difficulty_survive_toml_and_overrides_and_an_empty_value_clears_them(tmp_path):
    chosen = apply_overrides(ExperimentConfig(game_name="atari", game=AtariRules()), [parse_override(text) for text in ("game.mode=1", "game.difficulty=3")])
    assert (chosen.game.mode, chosen.game.difficulty) == (1, 3)
    save_experiment(chosen, tmp_path / "chosen.toml")
    assert load_experiment(tmp_path / "chosen.toml") == chosen
    cleared = apply_overrides(chosen, [parse_override(text) for text in ("game.mode=", "game.difficulty=")])
    assert (cleared.game.mode, cleared.game.difficulty) == (None, None)
    save_experiment(cleared, tmp_path / "cleared.toml")
    written = (tmp_path / "cleared.toml").read_text()
    assert "\nmode =" not in written and "\ndifficulty =" not in written and load_experiment(tmp_path / "cleared.toml") == cleared


def test_a_fork_may_change_the_mode_and_difficulty_but_not_the_title():
    parent = ExperimentConfig(game_name="atari", game=AtariRules(title="freeway"))
    require_forkable(parent, dataclasses.replace(parent, game=AtariRules(title="freeway", mode=3, difficulty=1)))
    assert locked_field_changes(parent, dataclasses.replace(parent, game=AtariRules(title="pong"))) == ["game.title"]


@pytest.mark.slow
def test_the_command_line_refuses_an_unavailable_mode_before_training(tmp_path):
    train = [sys.executable, str(Path(__file__).parents[1] / "scripts" / "train.py"), "--game", "atari", "--run-dir", str(tmp_path / "run")]
    result = subprocess.run([*train, "--set", "game.mode=9"], capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "error: Atari game 'pong' has no mode 9; available modes: 0, 1." in result.stderr
    assert "Traceback" not in result.stderr and not (tmp_path / "run").exists()


# --------------------------------------------------------------------------- shapes, preprocessing


def test_shapes_dtype_and_range():
    for rules, shape in [(REPEATABLE, (1, 84, 84)), (AtariRules(grayscale=False, screen_size=48), (3, 48, 48))]:
        env = make(rules)
        assert env.observation_shape == shape
        for observation in (env.reset(), env.step(2).observation):
            assert observation.shape == shape and observation.dtype == np.float32
            assert 0.0 <= observation.min() and observation.max() <= 1.0 and observation.max() > 0.2  # a picture, not zeros


def test_area_weights_average_the_pixels_each_output_pixel_covers():
    assert np.allclose(area_weights(4, 2), [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    assert np.allclose(area_weights(3, 2), [[2 / 3, 1 / 3, 0], [0, 1 / 3, 2 / 3]])  # the middle pixel is shared
    assert np.allclose(area_weights(210, 84).sum(axis=1), 1.0) and np.allclose(area_weights(5, 5), np.eye(5))


def test_shrinking_keeps_brightness_and_scales_to_one():
    shrink = ScreenShrinker(4, 6, 2)
    gray = np.array([[0, 0, 0, 255, 255, 255]] * 2 + [[255] * 6] * 2, dtype=np.uint8)
    assert np.allclose(shrink(gray), [[[0.0, 1.0], [1.0, 1.0]]])
    colour = np.zeros((4, 6, 3), dtype=np.uint8)
    colour[..., 1] = 51
    out = shrink(colour)
    assert out.shape == (3, 2, 2) and out.dtype == np.float32
    assert np.allclose(out[1], 0.2) and not out[0].any() and not out[2].any()


# --------------------------------------------------------------------------- stepping, with a scripted emulator


class ScriptedAle:
    """An emulator whose frame n shows brightness ``n`` everywhere, pays ``points[n]`` and ends at frame ``ends_at``."""

    def __init__(self, points=(), ends_at=None, lives=()):
        self.frame, self.acted = 0, []
        self._points, self._ends_at, self._lives = dict(points), ends_at, dict(lives)

    def getLegalActionSet(self):
        return [SimpleNamespace(name=name) for name in ALE_ACTION_NAMES]

    def getMinimalActionSet(self):
        return self.getLegalActionSet()[:3]

    def getScreenDims(self):
        return (4, 4)

    def getScreenGrayscale(self, buffer):
        buffer[...] = 0 if self.frame == 0 else 100 if self.frame % 2 else self.frame  # odd frames are brighter

    def act(self, action):
        self.frame += 1
        self.acted.append(action.name)
        return self._points.get(self.frame, 0)

    def game_over(self):
        return self._ends_at is not None and self.frame >= self._ends_at

    def lives(self):
        return next((count for frame, count in sorted(self._lives.items(), reverse=True) if self.frame >= frame), 3)


@pytest.fixture
def scripted(monkeypatch):
    """``make(rules, **script)``: an ``AtariEnv`` over a ``ScriptedAle``, and the emulator."""

    def build(rules=AtariRules(screen_size=36), reward=AtariReward(), **script):
        ale = ScriptedAle(**script)
        emulator = SimpleNamespace(ale=ale, reset=lambda seed=None: setattr(ale, "frame", 0))
        monkeypatch.setattr(atari_env, "make_emulator", lambda rules: emulator)
        return atari_env.AtariEnv(rules, reward), ale

    return build


def test_one_step_holds_the_action_for_frame_skip_frames_and_sums_their_points(scripted):
    env, ale = scripted(reward=AtariReward(clip=True), points={2: 3, 4: 5, 5: 100})
    env.reset()
    step = env.step(3)
    assert ale.acted == ["RIGHT"] * 4 and ale.frame == 4
    assert step.score == 8.0 and step.reward == 1.0 and step.event == "scored"  # clipped to the sign
    env, ale = scripted(AtariRules(screen_size=36, frame_skip=1), points={1: 2})
    env.reset()
    assert env.step(0).score == 2.0 and ale.frame == 1


def test_the_picture_is_the_brighter_of_the_last_two_frames(scripted):
    env, _ = scripted()
    env.reset()
    assert np.allclose(env.step(0).observation, 100 / 255)  # frames 3 (bright) and 4: the flickering object stays visible
    env, _ = scripted(AtariRules(screen_size=36, frame_skip=1))
    env.reset()
    env.step(0)
    assert np.allclose(env.step(0).observation, 2 / 255)  # frame 2 alone, not frame 1's 100


def test_the_real_emulator_advances_frame_skip_frames_per_step():
    for skip in (1, 4):
        env = make(AtariRules(sticky_actions=0.0, frame_skip=skip))
        play(env, [0], 5)
        assert env._ale.getEpisodeFrameNumber() == 5 * skip


def test_game_over_ends_the_hold_early_and_terminates(scripted):
    env, ale = scripted(ends_at=6, points={6: -1})
    env.reset()
    assert env.step(0).event == "playing"
    last = env.step(0)
    assert ale.frame == 6 and (last.terminated, last.truncated, last.event) == (True, False, "game_over")
    assert last.reward == -1.0


def test_lives_lost_are_events_and_can_cost_reward(scripted):
    env, _ = scripted(reward=AtariReward(life_lost=-2.0), lives={0: 3, 5: 2})
    env.reset()
    assert env.step(0).event == "playing"
    lost = env.step(0)
    assert (lost.event, lost.reward, lost.score) == ("life_lost", -2.0, 0.0)
    assert env.step(0).event == "playing"


def test_the_step_limit_truncates(scripted):
    env, _ = scripted(AtariRules(screen_size=36, max_episode_steps=2))
    env.reset()
    first, second = env.step(0), env.step(0)
    assert (first.truncated, first.event) == (False, "playing")
    assert (second.terminated, second.truncated, second.event) == (False, True, "timeout")
    env.reset()
    assert not env.step(0).truncated  # the count starts again


# --------------------------------------------------------------------------- actions


def test_action_count_with_and_without_the_full_action_space():
    full, own = make(), make(AtariRules(sticky_actions=0.0, full_action_space=False))
    assert (full.num_actions, own.num_actions) == (18, 6)
    assert full.action_labels == FULL_ACTION_LABELS == GAME.action_labels
    assert own.action_labels == ("nothing", "fire", "right", "left", "right + fire", "left + fire")  # Pong's own
    assert [action.name for action in full._actions] == list(ALE_ACTION_NAMES)
    assert len(set(FULL_ACTION_LABELS)) == 18 and action_labels(["UPFIRE", "NOOP"]) == ("up + fire", "nothing")


# --------------------------------------------------------------------------- seeding


def test_same_seed_same_game_when_actions_are_not_sticky():
    actions = [2, 5, 0, 3, 4, 1]
    a, b = play(make(), actions, 150, seed=7), play(make(), actions, 150, seed=7)
    assert [(s.reward, s.event, s.score) for s in a] == [(s.reward, s.event, s.score) for s in b]
    assert all(np.array_equal(x.observation, y.observation) for x, y in zip(a, b))
    assert not np.array_equal(a[0].observation, a[-1].observation)  # the game moved


def test_constructor_seed_is_used_by_the_first_plain_reset_and_sticky_games_repeat_per_seed():
    sticky, actions = AtariRules(sticky_actions=0.5), [2, 5, 5, 5, 2, 2, 0, 3]
    a, b = play(make(sticky, seed=3), actions, 300), play(make(sticky, seed=3), actions, 300)
    assert all(np.array_equal(x.observation, y.observation) for x, y in zip(a, b))
    other = play(make(sticky, seed=4), actions, 300)
    assert not all(np.array_equal(x.observation, y.observation) for x, y in zip(a, other))  # which frames stick depends on the seed


def test_without_sticky_actions_every_seed_is_the_same_game():
    """Pong itself has no randomness: only sticky actions make two games differ (so evaluate with them on)."""
    a, b = play(make(seed=3), [2, 5, 0, 3], 200), play(make(seed=4), [2, 5, 0, 3], 200)
    assert all(np.array_equal(x.observation, y.observation) for x, y in zip(a, b))


# --------------------------------------------------------------------------- reward, outcome, metrics, rules


def test_reward_clips_to_the_sign_scales_and_charges_lives():
    assert [atari_reward(AtariReward(clip=True), points, 0) for points in (0.0, 1.0, 500.0, -7.0)] == [0.0, 1.0, 1.0, -1.0]
    assert [atari_reward(AtariReward(), points, 0) for points in (0.0, 500.0, -7.0)] == [0.0, 500.0, -7.0]  # the default keeps the points
    assert atari_reward(AtariReward(clip=False, scale=0.01), 500.0, 0) == pytest.approx(5.0)
    assert atari_reward(AtariReward(clip=True, scale=0.5, life_lost=-1.0), 200.0, 2) == pytest.approx(0.5 - 2.0)


def test_score_is_the_raw_points_whatever_the_reward():
    steps = play(make(reward=AtariReward(scale=0.5)), [0], 400)  # doing nothing loses points in Pong
    assert steps[-1].score < 0 and steps[-1].score == pytest.approx(sum(s.reward for s in steps) / 0.5)
    assert {s.event for s in steps} == {"playing", "lost_points"}


@pytest.mark.parametrize(
    "points, lives_lost, game_over, timed_out, event",
    [
        (0.0, 0, False, False, "playing"),
        (10.0, 0, False, False, "scored"),
        (-1.0, 0, False, False, "lost_points"),
        (10.0, 1, False, False, "life_lost"),
        (0.0, 1, True, False, "game_over"),
        (5.0, 0, False, True, "timeout"),
    ],
)
def test_outcome_words(points, lives_lost, game_over, timed_out, event):
    assert step_event(points, lives_lost, game_over, timed_out) == event


def test_metrics_count_timeouts_lives_and_scoring_steps():
    def game(*events):
        return EpisodeRecord(score=0.0, length=len(events), total_reward=0.0, outcome=events[-1], events=events)

    stats = atari_statistics([game("playing", "scored", "life_lost", "scored", "game_over"), game("life_lost", "timeout")])
    assert stats == {"timeout_rate": 0.5, "mean_lives_lost": 1.0, "mean_scoring_steps": 1.0}
    assert set(stats) == {definition.key for definition in GAME.episode_metrics.definitions}


def test_rules_and_reward_validation():
    for bad in ({"frame_skip": 0}, {"sticky_actions": 1.0}, {"sticky_actions": -0.1}, {"max_episode_steps": 0}, {"screen_size": 35}):
        with pytest.raises(ValueError):
            AtariRules(**bad)
    with pytest.raises(ValueError):
        AtariReward(scale=0.0)


def test_view_is_the_full_colour_screen():
    env = make()
    env.reset()
    env.step(0)
    payload = env.view()
    assert (payload["width"], payload["height"]) == (160, 210)
    png = png_bytes(payload)
    assert png.startswith(b"\x89PNG") and struct.unpack(">II", png[16:24]) == (160, 210)


def test_finding_the_game_does_not_import_the_emulator():
    import subprocess
    import sys

    code = "import sys; import attentive_snake.games.atari; sys.exit(int('ale_py' in sys.modules or 'gymnasium' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --------------------------------------------------------------------------- fork locks, schema


def atari_config(**rules) -> ExperimentConfig:
    return ExperimentConfig(game_name="atari", game=AtariRules(**rules), reward=AtariReward(), model=ModelConfig(kind="cnn"))


def test_a_fork_must_keep_the_title_the_picture_and_the_action_space():
    locked = locked_fields("atari")
    assert {"game.title", "game.screen_size", "game.grayscale", "game.full_action_space", "env.history_length"} <= set(locked)
    assert not {"game.frame_skip", "game.sticky_actions", "game.max_episode_steps"} & set(locked)
    parent = atari_config()
    for change, key in [({"title": "venture"}, "game.title"), ({"screen_size": 64}, "game.screen_size"), ({"grayscale": False}, "game.grayscale"), ({"full_action_space": False}, "game.full_action_space")]:
        assert locked_field_changes(parent, atari_config(**change)) == [key]
        with pytest.raises(ValueError, match=key):
            require_forkable(parent, atari_config(**change))
    require_forkable(parent, atari_config(frame_skip=2, sticky_actions=0.0, max_episode_steps=100))
    require_forkable(parent, dataclasses.replace(parent, reward=AtariReward(clip=False, scale=0.1)))


def test_schema_offers_the_titles_and_only_the_kinds_that_read_pictures():
    schema = experiment_schema("atari")
    rules = {row["name"]: row for row in schema["sections"]["game"]}
    assert rules["title"]["choices"] == list(TITLE_KEYS) and rules["title"]["default"] == "pong"
    assert rules["full_action_space"]["default"] is True and rules["sticky_actions"]["default"] == 0.25
    assert [row["name"] for row in schema["sections"]["reward"]] == ["clip", "scale", "life_lost"]
    kinds = schema["model_kinds"]
    assert list(kinds) == IMAGE_KINDS and set(POLICY_KINDS) - set(kinds) == {"resnet", "mlp"}
    assert {kind for kind, row in kinds.items() if row["recommended"]} == {"cnn", "hybrid", "frame_attention"}
    for kind, row in kinds.items():
        assert row["description"] == KIND_INFO[kind].inputs["image"].description and "image" in row["observation_kinds"]
        assert not {"conv_layers", "kernel_size", "residual"} & set(row["fields"])
    assert "conv_channels" in kinds["cnn"]["fields"] and "kernel_size" in experiment_schema("snake")["model_kinds"]["cnn"]["fields"]
    assert set(experiment_schema("snake")["model_kinds"]) == set(POLICY_KINDS)  # every kind reads boards


# --------------------------------------------------------------------------- watching image models

WATCHED = ["cnn", "transformer", "hybrid", "temporal_transformer", "frame_attention", "lstm"]
SMALL = AtariRules(screen_size=48, max_episode_steps=6, full_action_space=False)  # 4 x 4 patches, a 2 x 2 feature map


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A tiny trained Pong run per watched kind (3 frames of 48 x 48 pixels)."""
    root = tmp_path_factory.mktemp("atari_runs")
    for kind in WATCHED:
        config = ExperimentConfig(
            game_name="atari",
            game=SMALL,
            reward=AtariReward(),
            model=ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=1, conv_channels=8),
            train=TrainConfig(total_env_steps=32, checkpoint_dir=root / kind, device="cpu"),
            algorithm_settings=PpoConfig(rollout_length=8, num_minibatches=2),
            env=EnvConfig(num_envs=2, history_length=3),
        )
        save_experiment(config, config_path(root / kind))
        env = make_vec_env(config)
        make_trainer(config, make_policy(config, env), env, NullLogger()).train()
    return root


@pytest.mark.parametrize("kind", WATCHED)
def test_watching_shows_the_screen_the_actions_and_the_declared_attention(runs, kind):
    session = WatchSession(runs / kind, seed=0)
    assert json.loads(json.dumps(session.info)) == session.info
    assert (session.info["view_kind"], session.info["width"], session.info["height"]) == ("image", 160, 210)
    assert session.info["observation_shape"] == [3, 48, 48] and session.info["element_names"] is None
    assert session.info["action_names"] == ["nothing", "fire", "right", "left", "right + fire", "left + fire"]  # the env's own, not the 18
    grid = {"transformer": (4, 4), "temporal_transformer": (4, 4), "hybrid": (2, 2)}.get(kind)
    frames = [session.reset(), *(session.step() for _ in range(7))]
    assert any(frame.event == "timeout" for frame in frames)
    for frame in frames:
        data = frame.to_dict()
        assert json.loads(json.dumps(data)) == data and data["png"] and data["element_weights"] is None
        if frame.done:
            continue
        assert len(data["action_values"]["values"]) == 6
        if grid:
            heatmap = np.array(data["heatmap"])
            assert heatmap.shape == grid and heatmap.min() >= 0.0 and heatmap.max() <= 1.0  # the tokens' rows and columns, not 48 x 48
            assert kind == "temporal_transformer" or heatmap.max() == pytest.approx(1.0)  # frames share one maximum
        elif kind == "frame_attention":
            assert len(data["frame_weights"]) == 3 and sum(data["frame_weights"]) == pytest.approx(1.0, abs=1e-5)
        else:
            assert data["heatmap"] is None and data["frame_weights"] is None


def test_the_observation_stacks_frames_oldest_first():
    config = dataclasses.replace(atari_config(sticky_actions=0.0), env=EnvConfig(history_length=4))
    env = make_env(config, seed=0)
    assert env.observation_shape == (4, 84, 84)
    first = env.reset()
    assert not first[:3].any() and first[3].any()  # earlier frames are empty at the start
    second = env.step(2).observation
    assert np.array_equal(second[2], first[3]) and second[3].any()
