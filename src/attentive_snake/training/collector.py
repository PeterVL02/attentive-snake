"""Running a policy in a ``VecEnv`` to fill a ``RolloutBuffer``."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from attentive_snake.env.types import EpisodeSummary, VecEnv
from attentive_snake.models.types import PolicyOutput
from attentive_snake.training.rollout import RolloutBuffer
from attentive_snake.training.truncation import TruncationBootstrap


@dataclass(frozen=True)
class CollectResult:
    """What a rollout produced besides the buffer contents."""

    bootstrap_value: Tensor  # (N,) value of the observation after the last step
    finished_episodes: tuple[EpisodeSummary, ...]


class RolloutCollector:
    """Steps a ``VecEnv`` with a policy and records the transitions.

    The collector owns the "current observation" of every environment between
    rollouts, so consecutive calls to ``collect`` continue the same episodes.

    The policy acts in ``eval()`` mode (dropout off) so the stored log-probs
    and values are those of the deterministic network; its previous
    train/eval mode is restored afterwards.

    Truncation: a step that ended its episode is ``done`` whether it was terminated or
    truncated. Given a ``TruncationBootstrap``, the reward of a truncated step also gets
    ``gamma * V(final observation)``, so its value target is bootstrapped; without one
    (the default) truncation is treated exactly like termination.
    """

    def __init__(
        self,
        env: VecEnv,
        buffer: RolloutBuffer,
        initial_observations: np.ndarray,
        truncation: TruncationBootstrap | None = None,
    ) -> None:
        if buffer.num_envs != env.num_envs:
            raise ValueError("Buffer and environment disagree on the number of environments.")
        self._env = env
        self._buffer = buffer
        self._device = buffer.observations.device
        self._observations = self._to_tensor(initial_observations)
        self._truncation = truncation

    @torch.no_grad()
    def collect(self, policy: nn.Module) -> CollectResult:
        was_training = policy.training
        policy.eval()
        try:
            return self._collect(policy)
        finally:
            policy.train(was_training)

    def _collect(self, policy: nn.Module) -> CollectResult:
        buffer = self._buffer
        finished: list[EpisodeSummary] = []
        for t in range(buffer.rollout_length):
            output = self._evaluate(policy, self._observations)
            distribution = Categorical(logits=output.logits)
            actions = distribution.sample()

            buffer.observations[t] = self._observations
            buffer.actions[t] = actions
            buffer.log_probs[t] = distribution.log_prob(actions)
            buffer.values[t] = output.value

            step = self._env.step(actions.cpu().numpy())
            rewards = self._to_tensor(step.rewards)
            if self._truncation is not None:
                rewards = self._truncation.rewards(rewards, step, lambda obs: self._evaluate(policy, obs).value)
            buffer.rewards[t] = rewards
            buffer.dones[t] = self._to_tensor(np.logical_or(step.terminated, step.truncated))
            self._observations = self._to_tensor(step.observations)
            finished.extend(step.finished_episodes)

        bootstrap_value = self._evaluate(policy, self._observations).value
        return CollectResult(bootstrap_value=bootstrap_value, finished_episodes=tuple(finished))

    def _evaluate(self, policy: nn.Module, observations: Tensor) -> PolicyOutput:
        output = policy(observations)
        batch = observations.shape[0]
        if output.logits.shape != (batch, self._env.num_actions):
            raise ValueError(f"Expected logits of shape {(batch, self._env.num_actions)}, got {tuple(output.logits.shape)}.")
        if output.value.shape != (batch,):
            raise ValueError(f"Expected value of shape {(batch,)}, got {tuple(output.value.shape)}.")
        return output

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self._device)
