"""Choosing an action from a policy (either head), and what the network says about each action."""

import torch
from torch import Tensor, nn

from attentive_snake.models.acting import ActionReadout, action_readout, greedy_actions, sampled_actions


@torch.no_grad()
def choose_action(policy: nn.Module, observation: Tensor, sample: bool) -> int:
    """The greedy action for a batch-of-one observation, or one sampled from the policy (greedy for Q-values)."""
    output = policy(observation)
    return int((sampled_actions(output) if sample else greedy_actions(output))[0])


@torch.no_grad()
def readout(policy: nn.Module, observation: Tensor) -> ActionReadout:
    """Action probabilities and state value (actor-critic) or Q-values, for a batch-of-one observation."""
    return action_readout(policy(observation))
