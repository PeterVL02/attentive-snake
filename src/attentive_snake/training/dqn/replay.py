"""The replay-buffer contract: what DQN stores and samples, so a prioritized buffer can replace the uniform one.

A collector ``add``s one ``Transitions`` per vector step (one transition per env). ``sample``
returns a ``ReplaySample`` with the buffer ``indices`` it drew and importance ``weights``
(all ones for uniform sampling); the learner scales each sample's loss by its weight and
reports the TD errors back with ``update_priorities`` (a no-op for uniform sampling).
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


@dataclass(frozen=True)
class ReplaySample:
    """A sampled batch on the training device, with where it came from and its importance weights."""

    observations: Tensor  # float32 (B, *obs_shape)
    actions: Tensor  # int64 (B,)
    rewards: Tensor  # float32 (B,)
    next_observations: Tensor  # float32 (B, *obs_shape)
    dones: Tensor  # float32 (B,)
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
