"""The game registry, and the proof that a game is drag-and-drop: a game living only in the test
tree (tests/dummy_games/walk) is found, configured, trained, described and watched through the
generic code without editing any file outside its own directory."""

import dataclasses
import sys
from pathlib import Path

import pytest

import attentive_snake.games as games_package
from attentive_snake.config import load_experiment, save_experiment
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.fork_rules import locked_fields, require_forkable
from attentive_snake.config.schema import experiment_schema
from attentive_snake.factory import make_env, make_policy, make_vec_env, observation_spec, make_trainer
from attentive_snake.games.registry import GameRegistryError, game_names, get_game, reload_games
from attentive_snake.games.snake import GAME as SNAKE
from attentive_snake.games.validation import spec_problems
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, Trainer
from attentive_snake.watch import WatchSession

DUMMY_GAMES = Path(__file__).resolve().parent / "dummy_games"


def _use_games_dir(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.setattr(games_package, "__path__", [*games_package.__path__, str(directory)])
    reload_games()


@pytest.fixture
def extra_games(monkeypatch: pytest.MonkeyPatch):
    """Add a games directory to the scan for one test; the registry and imports are restored afterwards."""
    added = []

    def add(directory: Path) -> None:
        added.append(directory)
        _use_games_dir(monkeypatch, directory)

    yield add
    monkeypatch.undo()
    reload_games()
    for name in [name for name in sys.modules if name.startswith(f"{games_package.__name__}.")]:
        if any(Path(getattr(sys.modules[name], "__file__", "") or "").is_relative_to(d) for d in added):
            del sys.modules[name]


def test_snake_is_registered() -> None:
    assert "snake" in game_names() and get_game("snake") is SNAKE
    assert (SNAKE.observation_kind, SNAKE.view_kind) == ("grid", "board")
    with pytest.raises(ValueError, match="unknown game 'chess'; known games: .*snake"):
        get_game("chess")


def test_spec_problems_are_reported() -> None:
    @dataclasses.dataclass(frozen=True)
    class Bad:
        name: str = "x"  # reserved, and without help

    broken = dataclasses.replace(SNAKE, name="Snake!", view_kind="movie", rules_type=Bad, locked_rules={"width": "?"})
    problems = " ".join(spec_problems(broken, "snake"))
    for fragment in ("must equal the package name", "a-z, 0-9", "view_kind 'movie'", "metadata['help']", "'name' (reserved)", "locked_rules names 'width'"):
        assert fragment in problems
    assert spec_problems(SNAKE, "snake") == []


def test_a_broken_game_package_is_named(extra_games, tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "__init__.py").write_text("NOT_GAME = 1\n")
    extra_games(tmp_path)
    with pytest.raises(GameRegistryError, match="attentive_snake.games.broken"):
        game_names()


def test_a_dropped_in_game_trains_and_is_watched_through_the_generic_code(extra_games, tmp_path: Path) -> None:
    extra_games(DUMMY_GAMES)
    assert "walk" in game_names() and "snake" in game_names()

    tables = {
        "game": {"name": "walk", "length": 4},
        "reward": {"goal": 2.0},
        "model": {"kind": "temporal_transformer", "d_model": 16, "num_heads": 2, "num_layers": 1, "mlp_ratio": 1},
        "train": {"total_env_steps": 64, "device": "cpu", "log_interval": 1, "checkpoint_dir": str(tmp_path / "run")},
        "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
        "env": {"num_envs": 4, "history_length": 2, "history_actions": True},
    }
    config = experiment_from_tables(tables)
    assert config.game_name == "walk" and config.game.length == 4 and config.reward.goal == 2.0

    env = make_vec_env(config)
    spec = observation_spec(config, env)
    assert (spec.kind, spec.shape, spec.base_channels, spec.action_channels) == ("grid", (8, 1, 4), 2, 2)
    policy = make_policy(config, env)
    history = make_trainer(config, policy, env, NullLogger()).train()
    assert len(history) == 2 and "mean_score" in history[-1]  # the walker reaches the goal in 12 steps or is cut off

    schema = experiment_schema("walk")
    assert [row["name"] for row in schema["sections"]["game"]] == ["length", "max_steps"]
    assert [row["name"] for row in schema["sections"]["reward"]] == ["goal", "step"]
    assert set(locked_fields("walk")) >= {"game.name", "game.length", "model.kind", "env.history_length"}
    with pytest.raises(ValueError, match="game.length"):
        require_forkable(config, dataclasses.replace(config, game=dataclasses.replace(config.game, length=5)))

    save_experiment(config, config_path(tmp_path / "run"))
    assert load_experiment(config_path(tmp_path / "run")) == config
    session = WatchSession(tmp_path / "run", seed=0)
    assert (session.info["game"], session.info["view_kind"], session.info["width"]) == ("walk", "board", 4)
    frames = [session.step().to_dict() for _ in range(5)]
    assert all(frame["view_kind"] == "board" and len(frame["snake"]) == 1 for frame in frames)
    assert make_env(config).view()["food"] == [3, 0]
