"""scripts/train.py with periodic evaluation and scripts/evaluate.py reproducing a recorded evaluation, on a tiny board."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from attentive_snake.config import load_experiment
from attentive_snake.factory import make_env, make_policy
from attentive_snake.training import save_checkpoint
from attentive_snake.training.checkpoint import checkpoint_path
from attentive_snake.evaluation.log import read_evaluations
from attentive_snake.run_dir import config_path, evaluations_path, list_checkpoints

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TINY = """
[game]
width = 4
height = 4
initial_length = 2
max_steps_without_food = 16
[model]
kind = "cnn"
conv_channels = 8
conv_layers = 1
[train]
total_env_steps = 192
rollout_length = 16
num_epochs = 1
num_minibatches = 2
log_interval = 1
checkpoint_interval = 100
device = "cpu"
[env]
num_envs = 4
"""  # no [evaluation]: like configs written before it existed


def script(name: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], capture_output=True, text=True, timeout=180)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finished tiny run with evaluation every 100 env steps (64 per update), 8 games each."""
    directory = tmp_path_factory.mktemp("evaluate")
    (directory / "tiny.toml").write_text(TINY)
    run_dir = directory / "snake" / "tiny"
    result = script("train.py", "--run-dir", run_dir, "--config", directory / "tiny.toml", "--eval-interval", 100, "--eval-games", 8)
    assert result.returncode == 0, result.stderr
    assert "evaluation steps=128 games=8" in result.stdout
    return run_dir


def test_training_records_evaluations_on_saved_checkpoints(run: Path):
    evaluations = read_evaluations(evaluations_path(run))
    assert [(e.env_steps, e.checkpoint, e.source) for e in evaluations] == [
        (128, "checkpoint_0000000128.pt", "training"),
        (192, "checkpoint_0000000192.pt", "training"),
    ]
    assert {path.name for path in list_checkpoints(run)} >= {e.checkpoint for e in evaluations}
    evaluation = load_experiment(config_path(run)).evaluation
    assert (evaluation.interval, evaluation.games, evaluation.seed) == (100, 8, 1000)
    assert evaluations[0].settings.games == 8 and set(evaluations[0].statistics) >= {"win_rate", "mean_score"}


def test_evaluate_reproduces_a_recorded_evaluation_exactly(run: Path):
    recorded = read_evaluations(evaluations_path(run))[0]
    result = script("evaluate.py", run, "--checkpoint", recorded.checkpoint, "--games", 8, "--seed", 1000)
    assert result.returncode == 0, result.stderr
    assert "Identical to the training evaluation" in result.stdout
    assert "win rate" in result.stdout and "N/A" in result.stdout  # an untrained snake wins nothing: no steps to win
    assert len(read_evaluations(evaluations_path(run))) == 2  # nothing saved without --save


def test_evaluate_with_other_settings_saves_a_manual_evaluation(run: Path):
    result = script("evaluate.py", run, "--games", 3, "--seed", 7, "--save")
    assert result.returncode == 0, result.stderr
    assert "No recorded evaluation" in result.stdout
    *_, saved = read_evaluations(evaluations_path(run))
    assert (saved.source, saved.checkpoint, saved.settings.games, saved.settings.seed) == ("manual", "checkpoint_0000000192.pt", 3, 7)


def test_a_run_whose_config_predates_evaluation_settings_is_evaluated_with_the_defaults(tmp_path: Path):
    run_dir = tmp_path / "old"
    run_dir.mkdir()
    config_path(run_dir).write_text(TINY)  # written before [evaluation] existed
    config = load_experiment(config_path(run_dir))
    policy = make_policy(config, make_env(config))
    save_checkpoint(checkpoint_path(run_dir, 64), policy, torch.optim.Adam(policy.parameters()), 64)
    result = script("evaluate.py", run_dir, "--games", 4)
    assert result.returncode == 0, result.stderr
    assert "checkpoint_0000000064.pt (64 env steps): 4 games, seed 1000" in result.stdout
    assert not evaluations_path(run_dir).exists()
