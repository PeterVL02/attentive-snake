"""Starting a new run from another run's checkpoint (a fork): weights and optimizer, but a fresh step counter."""

from pathlib import Path

from attentive_snake.training.trainer import Trainer


def fork_from_checkpoint(trainer: Trainer, path: Path) -> int:
    """Load policy, optimizer and learner state into ``trainer``, keep its step counter at 0; returns the parent's env steps.

    The new run has its own ``total_env_steps`` and schedules (learning rate, and for DQN
    exploration: epsilon starts again at ``epsilon_start``): the trainer sets the learning
    rate before every iteration from its own config, so a changed ``learning_rate`` (or
    ``anneal_lr``) takes effect from the first one. Adam's moment estimates are carried
    over in every case; they are running statistics of the gradients and do not depend on
    the learning rate. Call after building the trainer and before ``trainer.train()``.
    """
    return trainer.restore(path, keep_step_count=False)
