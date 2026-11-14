"""Starting a new run from another run's checkpoint (a fork): weights and optimizer, but a fresh step counter."""

from pathlib import Path

from attentive_snake.training.checkpoint import load_checkpoint
from attentive_snake.training.trainer import Trainer


def fork_from_checkpoint(trainer: Trainer, path: Path) -> int:
    """Load policy and optimizer state into ``trainer``, keep its step counter at 0; returns the parent's env steps.

    The new run has its own ``total_env_steps`` and learning-rate schedule: the
    trainer sets the learning rate before every update from its own config, so a
    changed ``learning_rate`` (or ``anneal_lr``) takes effect from the first
    update. Adam's moment estimates are carried over in every case; they are
    running statistics of the gradients and do not depend on the learning rate.
    Call after constructing the trainer and before ``trainer.train()``.
    """
    parent_steps = load_checkpoint(path, trainer.policy, trainer.optimizer, map_location=trainer.device)
    trainer.env_steps = 0
    return parent_steps
