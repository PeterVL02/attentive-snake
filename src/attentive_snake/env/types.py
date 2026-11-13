"""Contract between RL environments and whatever drives them (trainer, evaluation).

Environments speak numpy only; they know nothing about torch or models.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from attentive_snake.game.types import StepEvent


@dataclass(frozen=True)
class EpisodeSummary:
    """Statistics of one finished episode, for logging."""

    env_index: int
    score: int  # food eaten
    length: int  # steps taken
    total_reward: float
    final_event: StepEvent


@dataclass(frozen=True)
class EnvStep:
    """Result of stepping a single environment."""

    observation: np.ndarray  # float32, shape == observation_shape
    reward: float
    terminated: bool
    truncated: bool
    event: StepEvent


@dataclass(frozen=True)
class VecEnvStep:
    """Result of stepping N environments at once.

    Environments that finished are reset automatically, so ``observations[i]``
    is already the first observation of the next episode when
    ``terminated[i] or truncated[i]``.
    """

    observations: np.ndarray  # float32, (num_envs, *observation_shape)
    rewards: np.ndarray  # float32, (num_envs,)
    terminated: np.ndarray  # bool, (num_envs,)
    truncated: np.ndarray  # bool, (num_envs,)
    finished_episodes: tuple[EpisodeSummary, ...]


class VecEnv(Protocol):
    """What a trainer needs from a batch of parallel environments."""

    num_envs: int
    observation_shape: tuple[int, ...]  # (channels, height, width)
    num_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all environments; returns observations (num_envs, *observation_shape)."""
        ...

    def step(self, actions: np.ndarray) -> VecEnvStep:
        """Apply one integer action (an ``Action`` value) per environment."""
        ...
