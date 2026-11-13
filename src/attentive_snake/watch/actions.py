"""Choosing an action from a policy."""

import torch
from torch import Tensor, nn


@torch.no_grad()
def choose_action(policy: nn.Module, observation: Tensor, sample: bool) -> int:
    """Argmax of the logits for a (1, C, H, W) observation, or a sample from them."""
    logits = policy(observation).logits[0]
    if sample:
        return int(torch.distributions.Categorical(logits=logits).sample())
    return int(logits.argmax())
