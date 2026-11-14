"""A fixed-size ring buffer of transitions sampled uniformly (the ``ReplayBuffer`` of plain DQN)."""

import numpy as np
import torch

from attentive_snake.training.dqn.replay import ReplaySample, Transitions


class UniformReplayBuffer:
    """Holds the latest ``capacity`` transitions as float32 numpy arrays (observation and next observation each).

    Memory: ``2 * capacity * prod(observation_shape) * 4`` bytes plus a little per
    transition, allocated up front (the OS commits pages as they fill). Sampling draws slots
    with replacement from ``rng``, so a seeded generator gives reproducible batches.
    """

    def __init__(self, capacity: int, observation_shape: tuple[int, ...], rng: np.random.Generator) -> None:
        self._capacity = capacity
        self._rng = rng
        self._observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self._next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._next = 0  # slot written next
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transitions: Transitions) -> None:
        count = len(transitions.actions)
        slots = (self._next + np.arange(count)) % self._capacity
        self._observations[slots] = transitions.observations
        self._next_observations[slots] = transitions.next_observations
        self._actions[slots] = transitions.actions
        self._rewards[slots] = transitions.rewards
        self._dones[slots] = transitions.dones
        self._next = int((self._next + count) % self._capacity)
        self._size = min(self._size + count, self._capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplaySample:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer.")
        indices = self._rng.integers(0, self._size, size=batch_size)

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array[indices], device=device)

        return ReplaySample(
            observations=tensor(self._observations),
            actions=tensor(self._actions),
            rewards=tensor(self._rewards),
            next_observations=tensor(self._next_observations),
            dones=tensor(self._dones),
            weights=torch.ones(batch_size, dtype=torch.float32, device=device),
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Uniform sampling has no priorities."""
