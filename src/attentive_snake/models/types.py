"""Contract between policy networks and the trainers: what a policy's forward returns, per head.

Any ``nn.Module`` whose ``forward`` maps a batch of observations
``(batch, *observation_spec.shape)`` to the output of its head can be trained by the
algorithm that uses that head: ``PolicyOutput`` (actor-critic, PPO) or ``QValues``
(value-based, DQN). ``models.acting`` turns either into actions and displayable numbers.
"""

from typing import NamedTuple

from torch import Tensor


class PolicyOutput(NamedTuple):
    logits: Tensor  # (batch, num_actions), unnormalised action scores
    value: Tensor  # (batch,), state-value estimate


class QValues(NamedTuple):
    q_values: Tensor  # (batch, num_actions), expected discounted return of each action
