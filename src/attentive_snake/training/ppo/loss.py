"""The PPO loss for any action distribution (categorical or Gaussian) and the small statistics around it (pure functions)."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Distribution

from attentive_snake.training.ppo.value_loss import clipped_value_loss


@dataclass(frozen=True)
class LossTerms:
    """Loss components of one minibatch. ``total`` carries gradients; the rest are detached."""

    total: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


def ppo_loss(
    distribution: Distribution,
    values: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    old_values: Tensor | None = None,
) -> LossTerms:
    """Clipped-surrogate PPO loss (pure function).

    ``distribution`` is the current policy's (``models.action_distribution.policy_distribution``),
    batch shape (B,); ``actions`` are its samples ((B,) indices or (B, dimension) vectors); every
    other tensor is (B,). ``advantages`` are used as given (normalise them beforehand if desired).
    With ``old_values`` (the values at collection time) the value loss is clipped around them by ``clip_coef``.
    """
    if values.shape != returns.shape:
        # Guards against silent (B, 1) - (B,) broadcasting into a (B, B) loss.
        raise ValueError(f"values {tuple(values.shape)} and returns {tuple(returns.shape)} differ in shape.")
    log_probs = distribution.log_prob(actions)
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()

    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.max(unclipped, clipped).mean()
    value_loss = (
        0.5 * (values - returns).pow(2).mean() if old_values is None else clipped_value_loss(values, old_values, returns, clip_coef)
    )
    entropy = distribution.entropy().mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return LossTerms(
        total=total,
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        entropy=entropy.detach(),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Zero-mean, unit-std copy of ``x``. A single element maps to zero."""
    if x.numel() < 2:
        return x - x.mean()
    return (x - x.mean()) / (x.std() + eps)


def explained_variance(predictions: Tensor, targets: Tensor) -> float:
    """1 - Var[targets - predictions] / Var[targets]; NaN if targets are constant."""
    target_variance = targets.var()
    if target_variance == 0:
        return float("nan")
    return float(1.0 - (targets - predictions).var() / target_variance)
