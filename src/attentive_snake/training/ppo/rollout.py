"""Rollout storage and the computations that turn a rollout into training targets.

Indexing convention (shared by the collector and ``compute_gae``), for step
``t`` of environment ``i``:

* ``observations[t, i]`` - the observation the action was chosen from.
* ``actions[t, i]``, ``log_probs[t, i]``, ``values[t, i]`` - chosen action, its
  log-probability and V(observations[t, i]) under the acting policy.
* ``rewards[t, i]`` - reward received for that action.
* ``dones[t, i]`` - 1.0 if that action ended the episode (terminated or
  truncated). Because environments auto-reset, ``observations[t + 1, i]``
  then belongs to a *new* episode, so the value at ``t + 1`` must not be
  bootstrapped into step ``t``.
* ``next_observations[t, i]`` - only kept when a policy's auxiliary objective needs it
  (``None`` otherwise): what followed the action, the final observation of an episode
  that ended there (never the auto-reset one).
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RolloutBuffer:
    """Pre-allocated per-step storage of one rollout, shaped (steps, envs, ...)."""

    observations: Tensor  # float32 (T, N, *obs_shape)
    actions: Tensor  # int64 (T, N)
    log_probs: Tensor  # float32 (T, N)
    values: Tensor  # float32 (T, N)
    rewards: Tensor  # float32 (T, N)
    dones: Tensor  # float32 (T, N)
    next_observations: Tensor | None = None  # float32 (T, N, *obs_shape), when kept

    @classmethod
    def allocate(
        cls,
        rollout_length: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
        device: torch.device,
        keep_next_observations: bool = False,
    ) -> "RolloutBuffer":
        steps = (rollout_length, num_envs)

        def zeros(dtype: torch.dtype, extra: tuple[int, ...] = ()) -> Tensor:
            return torch.zeros(steps + extra, dtype=dtype, device=device)

        return cls(
            observations=zeros(torch.float32, tuple(observation_shape)),
            actions=zeros(torch.int64),
            log_probs=zeros(torch.float32),
            values=zeros(torch.float32),
            rewards=zeros(torch.float32),
            dones=zeros(torch.float32),
            next_observations=zeros(torch.float32, tuple(observation_shape)) if keep_next_observations else None,
        )

    @property
    def rollout_length(self) -> int:
        return self.actions.shape[0]

    @property
    def num_envs(self) -> int:
        return self.actions.shape[1]


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Generalised advantage estimation over a (T, N) rollout.

    Args:
        rewards, values, dones: (T, N), indexed as described in the module docstring.
        bootstrap_value: (N,) value of the observation following the last step.
        gamma, gae_lambda: Discount and GAE lambda.

    Returns:
        ``(advantages, returns)``, both (T, N), with ``returns = advantages + values``.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for t in reversed(range(num_steps)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


@dataclass(frozen=True)
class TrainingBatch:
    """A rollout flattened over (steps, envs) together with its training targets."""

    observations: Tensor  # (B, *obs_shape)
    actions: Tensor  # (B,)
    log_probs: Tensor  # (B,)
    values: Tensor  # (B,)
    advantages: Tensor  # (B,)
    returns: Tensor  # (B,)
    next_observations: Tensor | None = None  # (B, *obs_shape), when the rollout kept them

    def __len__(self) -> int:
        return self.actions.shape[0]

    def select(self, indices: Tensor) -> "TrainingBatch":
        return TrainingBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            log_probs=self.log_probs[indices],
            values=self.values[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            next_observations=None if self.next_observations is None else self.next_observations[indices],
        )


def flatten_rollout(buffer: RolloutBuffer, advantages: Tensor, returns: Tensor) -> TrainingBatch:
    """Merge the step and env dimensions of a rollout into one batch dimension."""
    return TrainingBatch(
        observations=buffer.observations.flatten(0, 1),
        actions=buffer.actions.flatten(),
        log_probs=buffer.log_probs.flatten(),
        values=buffer.values.flatten(),
        advantages=advantages.flatten(),
        returns=returns.flatten(),
        next_observations=None if buffer.next_observations is None else buffer.next_observations.flatten(0, 1),
    )
