"""Continuing a run: restoring a ``Trainer`` from a checkpoint."""

from pathlib import Path

from attentive_snake.training.checkpoint import load_checkpoint
from attentive_snake.training.trainer import Trainer


def resume_from_checkpoint(trainer: Trainer, path: Path) -> int:
    """Load policy and optimizer state into ``trainer`` and set its step counter; returns the env steps.

    Call after constructing the trainer (so the policy already sits on the
    training device) and before ``trainer.train()``.
    """
    trainer.env_steps = load_checkpoint(path, trainer.policy, trainer.optimizer, map_location=trainer.device)
    return trainer.env_steps
