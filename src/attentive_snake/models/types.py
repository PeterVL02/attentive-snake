"""Contract between policy networks and the trainer.

Any ``nn.Module`` whose ``forward`` maps a batch of observations
``(batch, *observation_spec.shape)`` to a ``PolicyOutput`` can be trained.
"""

from typing import NamedTuple

from torch import Tensor


class PolicyOutput(NamedTuple):
    logits: Tensor  # (batch, num_actions), unnormalised action scores
    value: Tensor  # (batch,), state-value estimate
