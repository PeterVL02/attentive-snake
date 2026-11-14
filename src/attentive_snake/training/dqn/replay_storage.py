"""The transitions of a replay buffer in a fixed-size ring: where they are written and how a batch is read back.

Shared by every ``ReplayBuffer`` (uniform and prioritized): they differ only in which slots they sample.
"""

import numpy as np
import torch

from attentive_snake.training.dqn.replay import ReplaySample, Transitions


class ReplayStorage:
    """Holds the latest ``capacity`` transitions as float32 numpy arrays (observation and next observation each).

    Memory: ``2 * capacity * prod(observation_shape) * 4`` bytes plus a little per
    transition, allocated up front (the OS commits pages as they fill).
    """

    def __init__(self, capacity: int, observation_shape: tuple[int, ...]) -> None:
        self._capacity = capacity
        self._observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self._next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._next = 0  # slot written next
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def write(self, transitions: Transitions) -> np.ndarray:
        """Store the transitions in the next slots, overwriting the oldest once full; returns the slots used."""
        count = len(transitions.actions)
        slots = (self._next + np.arange(count)) % self._capacity
        self._observations[slots] = transitions.observations
        self._next_observations[slots] = transitions.next_observations
        self._actions[slots] = transitions.actions
        self._rewards[slots] = transitions.rewards
        self._dones[slots] = transitions.dones
        self._next = int((self._next + count) % self._capacity)
        self._size = min(self._size + count, self._capacity)
        return slots

    def gather(self, indices: np.ndarray, weights: np.ndarray, device: torch.device) -> ReplaySample:
        """The transitions in ``indices`` (with their importance ``weights``) as one batch on ``device``."""

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array[indices], device=device)

        return ReplaySample(
            observations=tensor(self._observations),
            actions=tensor(self._actions),
            rewards=tensor(self._rewards),
            next_observations=tensor(self._next_observations),
            dones=tensor(self._dones),
            weights=torch.as_tensor(weights, dtype=torch.float32, device=device),
            indices=indices,
        )
