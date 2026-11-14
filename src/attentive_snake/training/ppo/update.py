"""The PPO update: the minibatch optimisation loop over one rollout."""

from dataclasses import dataclass

import torch
from torch import nn

from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.training.ppo.loss import LossTerms, explained_variance, normalize, ppo_loss
from attentive_snake.training.ppo.rollout import TrainingBatch


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


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: PpoConfig,
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
    config: PpoConfig,
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
    config: PpoConfig,
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
