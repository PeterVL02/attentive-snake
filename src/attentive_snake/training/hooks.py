"""What a ``Trainer`` tells its hooks after every update, so extra work (e.g. evaluation) stays out of the trainer."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class UpdateEnd:
    """The state right after one update (and after its logging and regular checkpoint).

    ``save_checkpoint()`` returns the checkpoint of the current weights at ``env_steps``,
    saving it first unless the trainer just did; ``None`` when the trainer has no
    checkpoint directory. A hook that changes the policy's mode must restore it.
    """

    previous_env_steps: int
    env_steps: int
    is_last: bool
    policy: nn.Module
    device: torch.device
    save_checkpoint: Callable[[], Path | None]


type UpdateHook = Callable[[UpdateEnd], None]
