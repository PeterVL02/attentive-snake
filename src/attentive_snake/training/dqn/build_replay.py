"""Chooses the replay buffer a DQN run uses from its settings."""

from collections.abc import Callable

import numpy as np

from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.dqn.n_step import Horizon
from attentive_snake.training.dqn.prioritized_replay import PrioritizedReplayBuffer
from attentive_snake.training.dqn.replay import ReplayBuffer
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer


def build_replay(
    settings: DqnConfig, observation_shape: tuple[int, ...], rng: np.random.Generator, beta: Callable[[], float], gamma: float
) -> ReplayBuffer:
    """Prioritized if ``settings.prioritized`` (``beta()`` is its current importance-sampling exponent), else uniform.

    Either samples targets that look ``settings.n_step`` steps ahead, discounted by ``gamma`` per step.
    """
    horizon = Horizon(settings.n_step, gamma)
    if not settings.prioritized:
        return UniformReplayBuffer(settings.buffer_size, observation_shape, rng, horizon)
    return PrioritizedReplayBuffer(
        settings.buffer_size, observation_shape, rng, settings.priority_alpha, settings.priority_epsilon, beta, horizon
    )
