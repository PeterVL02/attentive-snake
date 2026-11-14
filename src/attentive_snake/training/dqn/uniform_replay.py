"""Uniform sampling over a replay storage (the ``ReplayBuffer`` of plain DQN)."""

import numpy as np
import torch

from attentive_snake.training.dqn.replay import ReplaySample, Transitions
from attentive_snake.training.dqn.replay_storage import ReplayStorage


class UniformReplayBuffer:
    """Holds the latest ``capacity`` transitions (see ``ReplayStorage`` for the memory) and samples slots uniformly with replacement from ``rng``, so a seeded generator gives reproducible batches."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...], rng: np.random.Generator) -> None:
        self._storage = ReplayStorage(capacity, observation_shape)
        self._rng = rng

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transitions: Transitions) -> None:
        self._storage.write(transitions)

    def sample(self, batch_size: int, device: torch.device) -> ReplaySample:
        if len(self._storage) == 0:
            raise ValueError("cannot sample from an empty replay buffer.")
        indices = self._rng.integers(0, len(self._storage), size=batch_size)
        return self._storage.gather(indices, np.ones(batch_size, dtype=np.float32), device)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Uniform sampling has no priorities."""

    def statistics(self) -> dict[str, float]:
        """Uniform sampling has nothing to report."""
        return {}
