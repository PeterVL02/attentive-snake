"""Contract between a Q-value policy and DQN's loss: which value of a sampled transition the TD loss trains.

By default it is ``Q(s, a)`` of the action taken, read from the policy's own output. A policy
may instead supply the values to train with a method
``trained_values(observations, actions, next_observations) -> Tensor (B,) | None`` (``None``
= the default). The bootstrap targets do not change: they always come from the Q-values the
policy acts on. ``jepa`` with ``value_training = "actual"`` uses it to train its value
decoder on the embedding of the real next state.
"""

from torch import Tensor, nn


def trained_values(policy: nn.Module, observations: Tensor, actions: Tensor, next_observations: Tensor) -> Tensor | None:
    """The policy's own values to train for these transitions (B,), or ``None`` for ``Q(s, a)`` of its output."""
    supply = getattr(policy, "trained_values", None)
    return supply(observations, actions, next_observations) if callable(supply) else None
