"""One DQN gradient step on a replay sample: forward passes, loss, clipped optimiser step."""

from dataclasses import dataclass

import torch
from torch import nn

from attentive_snake.models.auxiliary import TransitionBatch
from attentive_snake.models.value_training import trained_values
from attentive_snake.training.auxiliary import AuxiliaryTraining
from attentive_snake.training.dqn.loss import DqnLoss, dqn_loss, td_loss
from attentive_snake.training.dqn.replay import ReplaySample
from attentive_snake.training.dqn.target import TargetNetwork


@dataclass(frozen=True)
class StepStats:
    loss: float
    q_mean: float
    td_errors: torch.Tensor  # (B,) detached, for ReplayBuffer.update_priorities


def dqn_gradient_step(
    policy: nn.Module,
    target: TargetNetwork,
    optimizer: torch.optim.Optimizer,
    sample: ReplaySample,
    gamma: float,
    double: bool,
    max_grad_norm: float,
    auxiliary: AuxiliaryTraining | None = None,
) -> StepStats:
    """Minimise the loss on ``sample`` once; the policy is in ``train()`` mode meanwhile and its mode is restored.

    With ``auxiliary`` (the policy's auxiliary objective) its weighted loss on the same
    transitions is added (every one counts: a stored next observation always followed its
    action, also when the episode ended there) and it is told after the optimizer step.
    """
    was_training = policy.training
    policy.train()
    try:
        terms = _loss(policy, target, sample, gamma, double)
        total = terms.loss if auxiliary is None else terms.loss + auxiliary.loss(_transitions(sample))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        if auxiliary is not None:
            auxiliary.after_step()
    finally:
        policy.train(was_training)
    return StepStats(loss=float(terms.loss.detach()), q_mean=float(terms.q_mean), td_errors=terms.td_errors)


def _transitions(sample: ReplaySample) -> TransitionBatch:
    return TransitionBatch(sample.observations, sample.actions, sample.next_observations, torch.ones_like(sample.rewards))


def _loss(policy: nn.Module, target: TargetNetwork, sample: ReplaySample, gamma: float, double: bool) -> DqnLoss:
    with torch.no_grad():
        next_q_target = target(sample.next_observations).q_values
        next_q_online = policy(sample.next_observations).q_values if double else None
    values = trained_values(policy, sample.observations, sample.actions, sample.next_observations)
    if values is not None:
        return td_loss(values, sample.rewards, sample.dones, next_q_target, next_q_online, gamma, sample.weights)
    return dqn_loss(
        q_values=policy(sample.observations).q_values,
        actions=sample.actions,
        rewards=sample.rewards,
        dones=sample.dones,
        next_q_target=next_q_target,
        next_q_online=next_q_online,
        gamma=gamma,
        weights=sample.weights,
    )
