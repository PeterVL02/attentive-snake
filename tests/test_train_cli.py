"""scripts/train.py end to end on a tiny board: descriptions, pruning, resuming and forking."""

import subprocess
import sys
from pathlib import Path

import pytest

from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path, list_checkpoints
from attentive_snake.run_metadata import Lineage, read_events

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
TINY = """
[game]
width = 4
height = 4
initial_length = 2
max_steps_without_food = 32
[model]
d_model = 16
num_heads = 2
num_layers = 1
mlp_ratio = 2
[train]
total_env_steps = 64
rollout_length = 8
num_epochs = 1
num_minibatches = 2
log_interval = 1
checkpoint_interval = 1
device = "cpu"
[env]
num_envs = 2
"""


def train(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=180)


def new_run(directory: Path) -> Path:
    """A finished tiny run in ``directory/parent`` that kept only its latest checkpoint."""
    (directory / "tiny.toml").write_text(TINY)
    run_dir = directory / "parent"
    result = train("--run-dir", run_dir, "--config", directory / "tiny.toml", "--no-keep-checkpoints", "--description", "first")
    assert result.returncode == 0, result.stderr
    return run_dir


@pytest.fixture(scope="module")
def parent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared by the tests that do not change it."""
    return new_run(tmp_path_factory.mktemp("cli"))


def test_new_run_records_description_and_prunes_checkpoints(parent: Path):
    assert [path.name for path in list_checkpoints(parent)] == ["checkpoint_0000000064.pt"]
    assert load_experiment(config_path(parent)).train.keep_checkpoints is False
    (event,) = read_events(parent)
    assert (event.action, event.description, event.env_steps_at_action) == ("start", "first", 0)


def test_resume_records_description_and_saves_keep_checkpoints(tmp_path: Path):
    parent = new_run(tmp_path)
    result = train("--resume", parent, "--steps", 96, "--keep-checkpoints", "--description", "longer")
    assert result.returncode == 0, result.stderr
    assert load_experiment(config_path(parent)).train.keep_checkpoints is True
    assert [path.name for path in list_checkpoints(parent)][-2:] == ["checkpoint_0000000080.pt", "checkpoint_0000000096.pt"]
    assert [(e.action, e.description, e.env_steps_at_action) for e in read_events(parent)] == [
        ("start", "first", 0),
        ("resume", "longer", 64),
    ]


def test_fork_with_changed_settings(parent: Path, tmp_path: Path):
    before = {path.name: path.stat().st_mtime_ns for path in parent.iterdir()}
    child = tmp_path / "child"
    result = train(
        "--run-dir", child, "--fork", parent, "--set", "train.learning_rate=1e-3", "--set", "reward.food=2",
        "--steps", 32, "--description", "lr up",
    )
    assert result.returncode == 0, result.stderr
    config = load_experiment(config_path(child))
    assert config.train.learning_rate == 1e-3 and config.reward.food == 2.0 and config.train.total_env_steps == 32
    assert config.model == load_experiment(config_path(parent)).model
    assert list_checkpoints(child)[-1].name == "checkpoint_0000000032.pt"  # its own counter, from 0
    (event,) = read_events(child)
    assert (event.action, event.description, event.env_steps_at_action) == ("fork", "lr up", 0)
    assert event.parent == Lineage("parent", "checkpoint_0000000064.pt", 64)
    assert {path.name: path.stat().st_mtime_ns for path in parent.iterdir()} == before  # the parent is untouched


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--set", "model.d_model=32"], "model.d_model"),
        (["--width", "5"], "game.width"),
        (["--set", "env.history_length=2"], "env.history_length"),
        (["--fork-checkpoint", "checkpoint_0000000001.pt"], "is not a checkpoint"),
    ],
)
def test_fork_rejections_write_nothing(parent: Path, tmp_path: Path, extra: list[str], message: str):
    child = tmp_path / "child"
    result = train("--run-dir", child, "--fork", parent, *extra)
    assert result.returncode == 2 and message in result.stderr
    assert not child.exists()


def test_fork_refuses_existing_runs_and_the_parent_itself(parent: Path):
    assert "--resume" in train("--run-dir", parent, "--fork", parent).stderr
    sibling = parent.parent / "sibling-run"
    assert train("--run-dir", sibling, "--fork", parent, "--steps", 16).returncode == 0
    refused = train("--run-dir", sibling, "--fork", parent)
    assert refused.returncode == 2 and "already contains checkpoints" in refused.stderr


def test_game_flag_picks_the_default_config_and_must_match_the_config(tmp_path: Path):
    (tmp_path / "tiny.toml").write_text(TINY)
    run_dir = tmp_path / "snake" / "g"
    result = train("--game", "snake", "--run-dir", run_dir, "--config", tmp_path / "tiny.toml", "--steps", 16)
    assert result.returncode == 0, result.stderr
    assert load_experiment(config_path(run_dir)).game_name == "snake"
    assert train("--game", "chess", "--run-dir", tmp_path / "x").returncode == 2  # not a known game
    (tmp_path / "other.toml").write_text('[game]\nname = "chess"\n')
    refused = train("--run-dir", tmp_path / "y", "--config", tmp_path / "other.toml")
    assert refused.returncode == 2 and "unknown game 'chess'" in refused.stderr
    resumed = train("--resume", run_dir, "--game", "snake")
    assert resumed.returncode == 2 and "--game" in resumed.stderr
