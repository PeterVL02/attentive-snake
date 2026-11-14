"""PPO as a ``Learner``: one iteration = collect a rollout -> GAE -> clipped-surrogate update."""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from attentive_snake.env.types import EpisodeSummary
from attentive_snake.training.algorithm import LearnerSetup
from attentive_snake.training.ppo.collector import RolloutCollector
from attentive_snake.training.ppo.rollout import RolloutBuffer, compute_gae, flatten_rollout
from attentive_snake.training.ppo.truncation import TruncationBootstrap
from attentive_snake.training.ppo.update import UpdateStats, ppo_update


class PpoLearner:
    """Owns PPO's optimizer, rollout buffer, collector and shuffling generator.

    Built after seeding, in this order (it fixes the random streams): optimizer,
    shuffling generator, buffer, env reset with ``train.seed``.
    """

    def __init__(self, setup: LearnerSetup) -> None:
        train, settings, env = setup.train, setup.settings, setup.env
        self.policy = setup.policy
        self._train, self._settings = train, settings
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=train.learning_rate, eps=1e-5)
        self._shuffle_generator = torch.Generator().manual_seed(train.seed)
        self._buffer = RolloutBuffer.allocate(settings.rollout_length, env.num_envs, tuple(env.observation_shape), setup.device)
        truncation = TruncationBootstrap(train.gamma) if train.bootstrap_truncated else None
        self._collector = RolloutCollector(env, self._buffer, env.reset(seed=train.seed), truncation)
        self._steps_per_iteration = settings.rollout_length * env.num_envs
        self._last: UpdateStats | None = None

    @property
    def steps_per_iteration(self) -> int:
        return self._steps_per_iteration

    def iterate(self, env_steps: int) -> tuple[EpisodeSummary, ...]:
        collected = self._collector.collect(self.policy)
        advantages, returns = compute_gae(
            self._buffer.rewards,
            self._buffer.values,
            self._buffer.dones,
            collected.bootstrap_value,
            self._train.gamma,
            self._settings.gae_lambda,
        )
        batch = flatten_rollout(self._buffer, advantages, returns)
        self._last = ppo_update(self.policy, self.optimizer, batch, self._settings, self._shuffle_generator)
        return collected.finished_episodes

    def report(self) -> dict[str, float]:
        """The statistics of the latest update."""
        return {} if self._last is None else {key: float(value) for key, value in dataclasses.asdict(self._last).items()}

    def extra_state(self) -> dict[str, Any]:
        return {}

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        pass
