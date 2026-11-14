"""Continuing a run: restoring a ``Trainer`` from one of its own checkpoints."""

from pathlib import Path

from attentive_snake.training.trainer import Trainer


def resume_from_checkpoint(trainer: Trainer, path: Path) -> int:
    """Load policy, optimizer and learner state into ``trainer`` and set its step counter; returns the env steps.

    Call after building the trainer (so the policy already sits on the training device)
    and before ``trainer.train()``. What the learner does not save (DQN's replay buffer)
    starts empty.
    """
    return trainer.restore(path, keep_step_count=True)
