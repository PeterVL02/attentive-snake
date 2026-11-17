"""Contract between policy networks and the trainers: what a policy's forward returns, per head.

Any ``nn.Module`` whose ``forward`` maps a batch of observations
``(batch, *observation_spec.shape)`` to the output of its head can be trained by the
algorithm that uses that head: ``PolicyOutput`` (actor-critic over discrete actions, PPO),
``GaussianPolicyOutput`` (actor-critic over continuous actions, PPO) or ``QValues``
(value-based, DQN). ``models.acting`` turns any of them into actions and displayable numbers,
``models.action_distribution`` an actor-critic output into the distribution PPO trains.
"""

from typing import NamedTuple

from torch import Tensor


class PolicyOutput(NamedTuple):
    logits: Tensor  # (batch, num_actions), unnormalised action scores
    value: Tensor  # (batch,), state-value estimate


class QValues(NamedTuple):
    q_values: Tensor  # (batch, num_actions), expected discounted return of each action


class GaussianPolicyOutput(NamedTuple):
    mean: Tensor  # (batch, dimension), the mean action (before clipping to the game's bounds)
    log_std: Tensor  # (batch, dimension), log standard deviation of each number (the same for every observation)
    value: Tensor  # (batch,), state-value estimate
