"""One DQN gradient step on a replay sample: forward passes, loss, clipped optimiser step."""

from dataclasses import dataclass

import torch
from torch import nn

from attentive_snake.models.auxiliary import TransitionBatch, TransitionSequences
from attentive_snake.models.value_training import trained_values
from attentive_snake.training.auxiliary import AuxiliaryTraining
from attentive_snake.training.dqn.loss import DqnLoss, ErrorLoss, dqn_loss, huber, td_loss
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
    double: bool,
    max_grad_norm: float,
    auxiliary: AuxiliaryTraining | None = None,
    error_loss: ErrorLoss = huber,
    sequences: TransitionSequences | None = None,
) -> StepStats:
    """Minimise the loss on ``sample`` once; the policy is in ``train()`` mode meanwhile and its mode is restored.

    The targets bootstrap from ``sample.bootstrap_observations`` (the end of each target window,
    scaled by ``sample.discounts``). With ``auxiliary`` (the policy's auxiliary objective) its
    weighted loss on the one-step transitions of the same sample is added (every one counts: a
    stored next observation always followed its action, also when the episode ended there) and
    it is told after the optimizer step. ``sequences`` (the steps that followed the sample's
    transitions, for an objective that reads more than one) are passed on to it.
    """
    was_training = policy.training
    policy.train()
    try:
        terms = _loss(policy, target, sample, double, error_loss)
        total = terms.loss if auxiliary is None else terms.loss + auxiliary.loss(_transitions(sample, sequences))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        if auxiliary is not None:
            auxiliary.after_step()
    finally:
        policy.train(was_training)
    return StepStats(loss=float(terms.loss.detach()), q_mean=float(terms.q_mean), td_errors=terms.td_errors)


def _transitions(sample: ReplaySample, sequences: TransitionSequences | None) -> TransitionBatch:
    return TransitionBatch(sample.observations, sample.actions, sample.next_observations, torch.ones_like(sample.rewards), sequences)


def _loss(policy: nn.Module, target: TargetNetwork, sample: ReplaySample, double: bool, error_loss: ErrorLoss) -> DqnLoss:
    with torch.no_grad():
        next_q_target = target(sample.bootstrap_observations).q_values
        next_q_online = policy(sample.bootstrap_observations).q_values if double else None
    values = trained_values(policy, sample.observations, sample.actions, sample.next_observations)  # the one-step next state
    if values is not None:
        return td_loss(values, sample.rewards, sample.dones, next_q_target, next_q_online, sample.discounts, sample.weights, error_loss)
    return dqn_loss(
        q_values=policy(sample.observations).q_values,
        actions=sample.actions,
        rewards=sample.rewards,
        dones=sample.dones,
        next_q_target=next_q_target,
        next_q_online=next_q_online,
        discounts=sample.discounts,
        weights=sample.weights,
        error_loss=error_loss,
    )
