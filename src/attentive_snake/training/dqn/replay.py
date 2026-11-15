"""The replay-buffer contract: what DQN stores and samples, so a prioritized buffer can replace the uniform one.

A collector ``add``s one ``Transitions`` per vector step (one transition per env). ``sample``
returns a ``ReplaySample`` with the buffer ``indices`` it drew and importance ``weights``
(all ones for uniform sampling); the learner scales each sample's loss by its weight and
reports the TD errors back with ``update_priorities`` (a no-op for uniform sampling). A prioritized
buffer takes what it needs of the training progress (the importance-sampling exponent beta) as a
callable at construction (``prioritized_replay.py``), so this contract stays free of schedules.

What is stored is always what happened in one step. What is sampled is a target window of up
to n steps starting there (``n_step.py``; n = 1: the step itself): ``rewards`` is the window's
discounted return, ``bootstrap_observations`` / ``dones`` are the observation after its last
step and whether nothing is bootstrapped from it, ``discounts`` is gamma ** (its steps).
``next_observations`` stays the observation one step after ``observations``.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Transitions:
    """One transition per env of a vector step, numpy on the CPU.

    ``next_observations[i]`` is the observation that followed the action, never the
    auto-reset start of the next episode; ``dones[i]`` is 1.0 when nothing is bootstrapped
    from it (the episode ended by its rules, or was cut and is treated as ended).
    """

    observations: np.ndarray  # float32 (N, *obs_shape)
    actions: np.ndarray  # int64 (N,)
    rewards: np.ndarray  # float32 (N,)
    next_observations: np.ndarray  # float32 (N, *obs_shape)
    dones: np.ndarray  # float32 (N,)
    ends: np.ndarray  # bool (N,), the episode ended with this step (by its rules or by a limit, bootstrapped or not)


@dataclass(frozen=True)
class ReplaySample:
    """A sampled batch on the training device, with where it came from and its importance weights."""

    observations: Tensor  # float32 (B, *obs_shape)
    actions: Tensor  # int64 (B,)
    rewards: Tensor  # float32 (B,), the discounted return of the target window (one step: that step's reward)
    next_observations: Tensor  # float32 (B, *obs_shape), one step after ``observations``
    bootstrap_observations: Tensor  # float32 (B, *obs_shape), after the window's last step (one step: ``next_observations``)
    dones: Tensor  # float32 (B,), 1.0 = nothing is bootstrapped from ``bootstrap_observations``
    discounts: Tensor  # float32 (B,), gamma ** (steps in the window): what the bootstrap value is scaled by
    weights: Tensor  # float32 (B,), importance-sampling weights (ones for uniform sampling)
    indices: np.ndarray  # int64 (B,), buffer slots, for update_priorities


class ReplayBuffer(Protocol):
    def __len__(self) -> int:
        """Transitions currently stored."""
        ...

    def add(self, transitions: Transitions) -> None: ...

    def sample(self, batch_size: int, device: torch.device) -> ReplaySample: ...

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Tell the buffer the absolute TD errors of the sampled slots (prioritized replay uses them)."""
        ...

    def statistics(self) -> dict[str, float]:
        """Numbers about the buffer's sampling for the learner's report (keys among the algorithm's metrics; none for uniform)."""
        ...
