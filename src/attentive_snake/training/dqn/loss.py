"""The DQN loss: Huber loss of the chosen actions' Q-values against (Double) DQN targets (pure function)."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DqnLoss:
    """``loss`` carries gradients; the rest are detached."""

    loss: Tensor  # scalar, importance-weighted mean Huber loss
    td_errors: Tensor  # (B,) Q(s, a) - target
    q_mean: Tensor  # scalar, mean Q(s, a) of the sampled actions


def dqn_targets(
    rewards: Tensor, dones: Tensor, next_q_target: Tensor, next_q_online: Tensor | None, gamma: float
) -> Tensor:
    """``r + gamma * (1 - done) * Q_target(s', a')`` with ``a' = argmax Q_online(s')`` (Double DQN) or ``argmax Q_target(s')`` (``next_q_online`` None)."""
    chooser = next_q_target if next_q_online is None else next_q_online
    next_actions = chooser.argmax(dim=-1, keepdim=True)
    next_values = next_q_target.gather(1, next_actions).squeeze(1)
    return rewards + gamma * (1.0 - dones) * next_values


def dqn_loss(
    q_values: Tensor,
    actions: Tensor,
    rewards: Tensor,
    dones: Tensor,
    next_q_target: Tensor,
    next_q_online: Tensor | None,
    gamma: float,
    weights: Tensor,
) -> DqnLoss:
    """``q_values``, ``next_q_*``: (B, A); the rest (B,). Targets carry no gradient; ``weights`` scale each sample's loss."""
    chosen = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    return td_loss(chosen, rewards, dones, next_q_target, next_q_online, gamma, weights)


def td_loss(
    chosen: Tensor,
    rewards: Tensor,
    dones: Tensor,
    next_q_target: Tensor,
    next_q_online: Tensor | None,
    gamma: float,
    weights: Tensor,
) -> DqnLoss:
    """The loss of ``dqn_loss`` for the values to train given directly (B,): usually Q(s, a), or what the policy supplies (``models.value_training``)."""
    with torch.no_grad():
        targets = dqn_targets(rewards, dones, next_q_target, next_q_online, gamma)
    per_sample = F.smooth_l1_loss(chosen, targets, reduction="none")
    return DqnLoss(
        loss=(weights * per_sample).mean(),
        td_errors=(chosen - targets).detach(),
        q_mean=chosen.detach().mean(),
    )
