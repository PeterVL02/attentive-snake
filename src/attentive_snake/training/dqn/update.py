"""One DQN gradient step on a replay sample: forward passes, loss, clipped optimiser step."""

from dataclasses import dataclass

import torch
from torch import nn

from attentive_snake.training.dqn.loss import DqnLoss, dqn_loss
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
) -> StepStats:
    """Minimise the loss on ``sample`` once; the policy is in ``train()`` mode meanwhile and its mode is restored."""
    was_training = policy.training
    policy.train()
    try:
        terms = _loss(policy, target, sample, gamma, double)
        optimizer.zero_grad(set_to_none=True)
        terms.loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
    finally:
        policy.train(was_training)
    return StepStats(loss=float(terms.loss.detach()), q_mean=float(terms.q_mean), td_errors=terms.td_errors)


def _loss(policy: nn.Module, target: TargetNetwork, sample: ReplaySample, gamma: float, double: bool) -> DqnLoss:
    with torch.no_grad():
        next_q_target = target(sample.next_observations).q_values
        next_q_online = policy(sample.next_observations).q_values if double else None
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
