"""The DQN loss: Huber (or squared-error) loss of the chosen actions' Q-values against (Double) DQN targets (pure functions)."""

from dataclasses import dataclass

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

type ErrorLoss = Callable[[Tensor, Tensor], Tensor]  # (values, targets) -> per-sample loss


def huber(values: Tensor, targets: Tensor) -> Tensor:
    """Per sample: ``0.5 * error ** 2`` up to an error of 1, ``|error| - 0.5`` beyond (large errors pull linearly)."""
    return F.smooth_l1_loss(values, targets, reduction="none")


def squared_error(values: Tensor, targets: Tensor) -> Tensor:
    """Per sample: ``error ** 2`` (large errors pull in proportion to their size)."""
    return F.mse_loss(values, targets, reduction="none")


ERROR_LOSSES: dict[str, ErrorLoss] = {"huber": huber, "mse": squared_error}  # by ``dqn.loss``


@dataclass(frozen=True)
class DqnLoss:
    """``loss`` carries gradients; the rest are detached."""

    loss: Tensor  # scalar, importance-weighted mean of the per-sample loss
    td_errors: Tensor  # (B,) Q(s, a) - target
    q_mean: Tensor  # scalar, mean Q(s, a) of the sampled actions


def dqn_targets(
    rewards: Tensor, dones: Tensor, next_q_target: Tensor, next_q_online: Tensor | None, discounts: Tensor | float
) -> Tensor:
    """``r + discount * (1 - done) * Q_target(s', a')`` with ``a' = argmax Q_online(s')`` (Double DQN) or ``argmax Q_target(s')`` (``next_q_online`` None).

    One step: ``r`` is the reward, ``s'`` the next state, ``discounts`` gamma. n steps: ``r`` is the
    discounted return of the n steps, ``s'`` the state after them, ``discounts`` (B,) gamma ** n.
    """
    chooser = next_q_target if next_q_online is None else next_q_online
    next_actions = chooser.argmax(dim=-1, keepdim=True)
    next_values = next_q_target.gather(1, next_actions).squeeze(1)
    return rewards + discounts * (1.0 - dones) * next_values


def dqn_loss(
    q_values: Tensor,
    actions: Tensor,
    rewards: Tensor,
    dones: Tensor,
    next_q_target: Tensor,
    next_q_online: Tensor | None,
    discounts: Tensor | float,
    weights: Tensor,
    error_loss: ErrorLoss = huber,
) -> DqnLoss:
    """``q_values``, ``next_q_*``: (B, A); the rest (B,). Targets carry no gradient; ``weights`` scale each sample's loss."""
    chosen = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    return td_loss(chosen, rewards, dones, next_q_target, next_q_online, discounts, weights, error_loss)


def td_loss(
    chosen: Tensor,
    rewards: Tensor,
    dones: Tensor,
    next_q_target: Tensor,
    next_q_online: Tensor | None,
    discounts: Tensor | float,
    weights: Tensor,
    error_loss: ErrorLoss = huber,
) -> DqnLoss:
    """The loss of ``dqn_loss`` for the values to train given directly (B,): usually Q(s, a), or what the policy supplies (``models.value_training``)."""
    with torch.no_grad():
        targets = dqn_targets(rewards, dones, next_q_target, next_q_online, discounts)
    per_sample = error_loss(chosen, targets)
    return DqnLoss(
        loss=(weights * per_sample).mean(),
        td_errors=(chosen - targets).detach(),
        q_mean=chosen.detach().mean(),
    )
