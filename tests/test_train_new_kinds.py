"""resnet and lstm trained through scripts/train.py with both algorithms: checkpoints load, resume continues."""

from pathlib import Path

import pytest

from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path, list_checkpoints
from attentive_snake.training.checkpoint import read_checkpoint
from attentive_snake.training.metrics import read_metrics
from test_train_cli import TINY, train

DQN = ["--set", "train.algorithm=dqn", "--set", "dqn.batch_size=8", "--set", "dqn.learning_starts=16", "--set", "dqn.buffer_size=64"]
HEADS = {"ppo": "net.value_head.linear.weight", "dqn": "net.q_head.linear.weight"}
TRUNK = {"resnet": "net.encoder.blocks.0.", "lstm": "net.recurrent.lstm.weight_ih_l0"}


@pytest.mark.parametrize("kind, algorithm, resume", [("resnet", "ppo", False), ("resnet", "dqn", True), ("lstm", "ppo", True), ("lstm", "dqn", False)])
def test_kind_trains_saves_and_resumes(tmp_path: Path, kind: str, algorithm: str, resume: bool):
    (tmp_path / "tiny.toml").write_text(TINY)
    run_dir = tmp_path / "run"
    history = ["--set", "env.history_length=3", "--set", "env.history_actions=true"] if kind == "lstm" else []
    result = train(
        "--run-dir", run_dir, "--config", tmp_path / "tiny.toml", "--set", f"model.kind={kind}", "--set", "model.conv_channels=8",
        "--set", "model.conv_layers=2", *history, *(DQN if algorithm == "dqn" else []),
    )
    assert result.returncode == 0, result.stderr
    config = load_experiment(config_path(run_dir))
    assert (config.model.kind, config.train.algorithm) == (kind, algorithm)
    policy = read_checkpoint(list_checkpoints(run_dir)[-1])["policy"]
    assert HEADS[algorithm] in policy and any(key.startswith(TRUNK[kind]) for key in policy)
    assert read_metrics(run_dir / "metrics.jsonl")[-1]["env_steps"] == 64
    if resume:
        assert train("--resume", run_dir, "--steps", 96).returncode == 0
        assert list_checkpoints(run_dir)[-1].name == "checkpoint_0000000096.pt"
