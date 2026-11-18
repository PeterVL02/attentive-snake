"""PPO's clipped value loss (pure function)."""

import torch
from torch import Tensor


def clipped_value_loss(values: Tensor, old_values: Tensor, returns: Tensor, clip_range: float) -> Tensor:
    """``0.5 * mean(max(unclipped error^2, clipped error^2))``; the clipped prediction stays within ``clip_range`` of ``old_values``."""
    clipped = old_values + torch.clamp(values - old_values, -clip_range, clip_range)
    return 0.5 * torch.max((values - returns).pow(2), (clipped - returns).pow(2)).mean()
