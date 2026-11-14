"""Proportional prioritized replay (Schaul et al., 2016, Algorithm 1) over a replay storage."""

from collections.abc import Callable

import numpy as np
import torch

from attentive_snake.training.dqn.replay import ReplaySample, Transitions
from attentive_snake.training.dqn.replay_storage import ReplayStorage
from attentive_snake.training.dqn.sum_tree import SumTree


class PrioritizedReplayBuffer:
    """Samples slots with probability proportional to ``priority = (|TD error| + epsilon) ** alpha``.

    - A new transition gets the largest priority seen so far (1 before any update), so it is replayed at least once.
    - ``sample`` is stratified: the priority mass is cut into ``batch_size`` equal segments with one draw in each.
    - Importance weights ``(size * P(i)) ** -beta`` are divided by the batch maximum; ``beta()`` is asked at every
      ``sample``, so whoever owns the training progress decides how it is annealed.
    - Priorities belong to slots: overwriting a slot resets its priority; wrapping around needs no extra care.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        rng: np.random.Generator,
        alpha: float,
        epsilon: float,
        beta: Callable[[], float],
    ) -> None:
        self._storage = ReplayStorage(capacity, observation_shape)
        self._tree = SumTree(capacity)
        self._rng, self._alpha, self._epsilon, self._beta = rng, alpha, epsilon, beta
        self._max_priority = 1.0  # the running maximum of every priority ever set (it never decays)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transitions: Transitions) -> None:
        slots = self._storage.write(transitions)
        self._tree.set(slots, np.full(len(slots), self._max_priority))

    def sample(self, batch_size: int, device: torch.device) -> ReplaySample:
        if len(self._storage) == 0:
            raise ValueError("cannot sample from an empty replay buffer.")
        total = self._tree.total
        segment = total / batch_size
        prefix_sums = (np.arange(batch_size) + self._rng.random(batch_size)) * segment
        indices = self._tree.find(np.minimum(prefix_sums, np.nextafter(total, 0.0)))
        probabilities = self._tree[indices] / total
        weights = (len(self._storage) * probabilities) ** -self._beta()
        return self._storage.gather(indices, weights / weights.max(), device)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Give the slots ``(|td| + epsilon) ** alpha``; a TD error of 0 leaves them samplable (epsilon > 0)."""
        priorities = (np.abs(td_errors) + self._epsilon) ** self._alpha
        self._tree.set(indices, priorities)
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def statistics(self) -> dict[str, float]:
        """The current beta, and the mean and largest priority."""
        return {
            "priority_beta": self._beta(),
            "priority_mean": self._tree.total / len(self._storage) if len(self._storage) else 0.0,
            "priority_max": self._max_priority,
        }
