"""Choosing an action from a policy (either head), and what the network says about each action."""

import numpy as np
import torch
from torch import Tensor, nn

from attentive_snake.models.acting import ActionReadout, action_readout, greedy_actions, sampled_actions


@torch.no_grad()
def choose_action(policy: nn.Module, observation: Tensor, sample: bool) -> int | np.ndarray:
    """The greedy action for a batch-of-one observation, or one sampled from the policy (greedy for Q-values):
    an index, or an action vector (continuous actions; the env clips it to the game's bounds)."""
    output = policy(observation)
    action = (sampled_actions(output) if sample else greedy_actions(output))[0]
    return int(action) if action.ndim == 0 else action.cpu().numpy()


@torch.no_grad()
def readout(policy: nn.Module, observation: Tensor) -> ActionReadout:
    """Action probabilities and state value (actor-critic), the mean action with its spread and the state value
    (continuous actor-critic) or Q-values, for a batch-of-one observation."""
    return action_readout(policy(observation))
