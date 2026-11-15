"""The transitions of a replay buffer in a fixed-size ring: where they are written and how a batch is read back.

Shared by every ``ReplayBuffer`` (uniform and prioritized): they differ only in which slots they sample.
The slots hold one-step transitions; a batch is read back as target windows of the storage's
``Horizon`` (``n_step.py``), which for n = 1 are the stored transitions themselves.
"""

import numpy as np
import torch

from attentive_snake.training.dqn.n_step import Horizon, n_step_windows
from attentive_snake.training.dqn.replay import ReplaySample, Transitions


class ReplayStorage:
    """Holds the latest ``capacity`` transitions as float32 numpy arrays (observation and next observation each).

    Memory: ``2 * capacity * prod(observation_shape) * 4`` bytes plus a little per
    transition, allocated up front (the OS commits pages as they fill).

    With ``horizon.n`` above 1 every ``write`` must bring the same number of transitions (one
    per env of a vector step, in env order): that is how a later step of an env is found.
    """

    def __init__(self, capacity: int, observation_shape: tuple[int, ...], horizon: Horizon) -> None:
        self._capacity = capacity
        self._horizon = horizon
        self._observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self._next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._ends = np.zeros(capacity, dtype=bool)
        self._stride = 1  # transitions per write: the slot distance between two steps of one env
        self._next = 0  # slot written next
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def write(self, transitions: Transitions) -> np.ndarray:
        """Store the transitions in the next slots, overwriting the oldest once full; returns the slots used."""
        count = len(transitions.actions)
        if self._horizon.n > 1 and self._size > 0 and count != self._stride:
            raise ValueError(f"n-step replay needs the same number of transitions in every write ({self._stride}), got {count}.")
        self._stride = count
        slots = (self._next + np.arange(count)) % self._capacity
        self._observations[slots] = transitions.observations
        self._next_observations[slots] = transitions.next_observations
        self._actions[slots] = transitions.actions
        self._rewards[slots] = transitions.rewards
        self._dones[slots] = transitions.dones
        self._ends[slots] = transitions.ends
        self._next = int((self._next + count) % self._capacity)
        self._size = min(self._size + count, self._capacity)
        return slots

    def gather(self, indices: np.ndarray, weights: np.ndarray, device: torch.device) -> ReplaySample:
        """The target windows starting at ``indices`` (with their importance ``weights``) as one batch on ``device``."""

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, device=device)

        later_steps = ((self._next - 1 - indices) % self._capacity) // self._stride
        windows = n_step_windows(self._rewards, self._ends, indices, later_steps, self._stride, self._horizon)
        following = tensor(self._next_observations[indices])
        one_step = self._horizon.n == 1
        return ReplaySample(
            observations=tensor(self._observations[indices]),
            actions=tensor(self._actions[indices]),
            rewards=tensor(windows.returns),
            next_observations=following,
            bootstrap_observations=following if one_step else tensor(self._next_observations[windows.last]),
            dones=tensor(self._dones[windows.last]),
            discounts=tensor(windows.discounts),
            weights=torch.as_tensor(weights, dtype=torch.float32, device=device),
            indices=indices,
        )
