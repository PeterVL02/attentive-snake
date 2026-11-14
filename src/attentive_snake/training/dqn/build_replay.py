"""Chooses the replay buffer a DQN run uses from its settings."""

from collections.abc import Callable

import numpy as np

from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.dqn.prioritized_replay import PrioritizedReplayBuffer
from attentive_snake.training.dqn.replay import ReplayBuffer
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer


def build_replay(
    settings: DqnConfig, observation_shape: tuple[int, ...], rng: np.random.Generator, beta: Callable[[], float]
) -> ReplayBuffer:
    """Prioritized if ``settings.prioritized`` (``beta()`` is its current importance-sampling exponent), else uniform."""
    if not settings.prioritized:
        return UniformReplayBuffer(settings.buffer_size, observation_shape, rng)
    return PrioritizedReplayBuffer(
        settings.buffer_size, observation_shape, rng, settings.priority_alpha, settings.priority_epsilon, beta
    )
