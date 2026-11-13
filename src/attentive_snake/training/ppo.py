"""The PPO update: clipped-surrogate loss and the minibatch optimisation loop."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from attentive_snake.training.config import TrainConfig
from attentive_snake.training.rollout import TrainingBatch


@dataclass(frozen=True)
class LossTerms:
    """Loss components of one minibatch. ``total`` carries gradients; the rest are detached."""

    total: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True)
class UpdateStats:
    """Averages over all minibatches of one update, plus explained variance of the rollout."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    epochs_completed: int


def ppo_loss(
    logits: Tensor,
    values: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
) -> LossTerms:
    """Clipped-surrogate PPO loss for a categorical policy (pure function).

    ``logits`` is (B, A); every other tensor is (B,). ``advantages`` are used
    as given (normalise them beforehand if desired).
    """
    if values.shape != returns.shape:
        # Guards against silent (B, 1) - (B,) broadcasting into a (B, B) loss.
        raise ValueError(f"values {tuple(values.shape)} and returns {tuple(returns.shape)} differ in shape.")
    distribution = Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()

    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.max(unclipped, clipped).mean()
    value_loss = 0.5 * (values - returns).pow(2).mean()
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


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: TrainConfig,
    generator: torch.Generator | None = None,
) -> UpdateStats:
    """Run ``config.num_epochs`` epochs of shuffled minibatch updates on ``batch``.

    The policy is put in ``train()`` mode for the update and its previous mode
    is restored afterwards. ``generator`` (a CPU generator) controls the
    minibatch shuffling; ``None`` uses torch's global RNG.
    """
    was_training = policy.training
    policy.train()
    try:
        return _run_epochs(policy, optimizer, batch, config, generator)
    finally:
        policy.train(was_training)


def _run_epochs(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: TrainConfig,
    generator: torch.Generator | None,
) -> UpdateStats:
    totals = torch.zeros(5, device=batch.actions.device)
    num_minibatches_run = 0
    epochs_completed = 0
    for _ in range(config.num_epochs):
        permutation = torch.randperm(len(batch), generator=generator).to(batch.actions.device)
        epoch_kl = torch.zeros((), device=batch.actions.device)
        chunks = torch.tensor_split(permutation, config.num_minibatches)
        for indices in chunks:
            terms = _minibatch_step(policy, optimizer, batch.select(indices), config)
            totals += torch.stack(
                [terms.policy_loss, terms.value_loss, terms.entropy, terms.approx_kl, terms.clip_fraction]
            )
            epoch_kl += terms.approx_kl
            num_minibatches_run += 1
        epochs_completed += 1
        if config.target_kl is not None and float(epoch_kl) / len(chunks) > config.target_kl:
            break

    means = (totals / num_minibatches_run).tolist()
    return UpdateStats(
        policy_loss=means[0],
        value_loss=means[1],
        entropy=means[2],
        approx_kl=means[3],
        clip_fraction=means[4],
        explained_variance=explained_variance(batch.values, batch.returns),
        epochs_completed=epochs_completed,
    )


def _minibatch_step(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    minibatch: TrainingBatch,
    config: TrainConfig,
) -> LossTerms:
    output = policy(minibatch.observations)
    advantages = normalize(minibatch.advantages) if config.normalize_advantages else minibatch.advantages
    terms = ppo_loss(
        logits=output.logits,
        values=output.value,
        actions=minibatch.actions,
        old_log_probs=minibatch.log_probs,
        advantages=advantages,
        returns=minibatch.returns,
        clip_coef=config.clip_coef,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
    )
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
    optimizer.step()
    return terms
