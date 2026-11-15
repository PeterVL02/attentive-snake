"""The transitions of a replay buffer in a fixed-size ring: where they are written and how a batch is read back.

Shared by every ``ReplayBuffer`` (uniform and prioritized): they differ only in which slots they sample.
The slots hold one-step transitions; a batch is read back as target windows of the storage's
``Horizon`` (``n_step.py``), which for n = 1 are the stored transitions themselves; ``sequences``
reads the steps that followed given slots one by one.
"""

import numpy as np
import torch

from attentive_snake.models.auxiliary import TransitionSequences
from attentive_snake.training.dqn.n_step import Horizon, n_step_windows
from attentive_snake.training.dqn.replay import ReplaySample, Transitions
from attentive_snake.training.slot_sequences import later_steps_in_ring, slot_sequences


class ReplayStorage:
    """Holds the latest ``capacity`` transitions as float32 numpy arrays (observation and next observation each).

    Memory: ``2 * capacity * prod(observation_shape) * 4`` bytes plus a little per
    transition, allocated up front (the OS commits pages as they fill).

    With ``horizon.n`` above 1 every ``write`` must bring the same number of transitions (one
    per env of a vector step, in env order): that is how a later step of an env is found.
    ``sequences`` needs the same and refuses a storage that was written otherwise.
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
        self._even_writes = True  # every write so far brought ``_stride`` transitions
        self._next = 0  # slot written next
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def write(self, transitions: Transitions) -> np.ndarray:
        """Store the transitions in the next slots, overwriting the oldest once full; returns the slots used."""
        count = len(transitions.actions)
        if self._horizon.n > 1 and self._size > 0 and count != self._stride:
            raise ValueError(f"n-step replay needs the same number of transitions in every write ({self._stride}), got {count}.")
        self._even_writes &= self._size == 0 or count == self._stride
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

        windows = n_step_windows(self._rewards, self._ends, indices, self._later_steps(indices), self._stride, self._horizon)
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

    def sequences(self, indices: np.ndarray, length: int, device: torch.device) -> TransitionSequences:
        """The ``length`` stored steps of the same env from each slot of ``indices`` on, as one batch on ``device``.

        Column 0 is the slot's own transition (always valid: a stored next observation followed
        its action); a later column is valid while the episode went on and the step is stored.
        """
        if not self._even_writes:
            raise ValueError("sequences need the same number of transitions in every write (one per env).")
        found = slot_sequences(self._ends, indices, self._later_steps(indices), self._stride, length)
        return TransitionSequences(
            actions=torch.as_tensor(self._actions[found.slots], device=device),
            next_observations=torch.as_tensor(self._next_observations[found.slots], device=device),
            valid=torch.as_tensor(found.present.astype(np.float32), device=device),
            terminal=torch.as_tensor(self._dones[found.slots], device=device),
        )

    def _later_steps(self, indices: np.ndarray) -> np.ndarray:
        return later_steps_in_ring(indices, self._next, self._capacity, self._stride)
