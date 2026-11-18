"""PPO as a ``Learner``: one iteration = collect a rollout -> GAE -> clipped-surrogate update."""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from attentive_snake.env.action_space import action_space_of
from attentive_snake.env.types import EpisodeSummary
from attentive_snake.training.algorithm import LearnerSetup
from attentive_snake.training.auxiliary import AuxiliaryTraining
from attentive_snake.training.optimizer import build_optimizer
from attentive_snake.training.ppo.collector import RolloutCollector
from attentive_snake.training.ppo.return_normalizer import ReturnNormalizer
from attentive_snake.training.ppo.rollout import RolloutBuffer, compute_gae, flatten_rollout
from attentive_snake.training.ppo.sequences import rollout_sequences
from attentive_snake.training.ppo.truncation import TruncationBootstrap
from attentive_snake.training.ppo.update import UpdateStats, ppo_update

_RETURN_STATISTICS_KEY = "return_statistics"  # where the return normalizer's state sits in extra_state


class PpoLearner:
    """Owns PPO's optimizer, rollout buffer, collector and shuffling generator.

    Built after seeding, in this order (it fixes the random streams): optimizer,
    shuffling generator, buffer, env reset with ``train.seed``.

    With ``normalize_observations`` the rollout keeps raw observations and the policy's statistics
    learn from them only after the update: a rollout and the update on it read the observations
    alike, so the stored log-probabilities and values stay those of the network being updated.
    """

    def __init__(self, setup: LearnerSetup) -> None:
        train, settings, env = setup.train, setup.settings, setup.env
        self.policy = setup.policy
        self._train, self._settings = train, settings
        self.optimizer = build_optimizer(self.policy, train, eps=1e-5)
        self._shuffle_generator = torch.Generator().manual_seed(train.seed)
        self._auxiliary = AuxiliaryTraining.of(self.policy)
        self._buffer = RolloutBuffer.allocate(
            settings.rollout_length,
            env.num_envs,
            tuple(env.observation_shape),
            setup.device,
            self._auxiliary is not None,
            action_space_of(env),
        )
        truncation = TruncationBootstrap(train.gamma) if train.bootstrap_truncated else None
        self._return_normalizer = ReturnNormalizer(env.num_envs, train.gamma, setup.device) if settings.normalize_returns else None
        self._collector = RolloutCollector(env, self._buffer, env.reset(seed=train.seed), truncation, self._return_normalizer)
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
        steps = 1 if self._auxiliary is None else self._auxiliary.sequence_length
        sequences = rollout_sequences(self._buffer, steps) if steps > 1 else None  # K copies of the next observations
        batch = flatten_rollout(self._buffer, advantages, returns, sequences)
        self._last = ppo_update(self.policy, self.optimizer, batch, self._settings, self._shuffle_generator, self._auxiliary)
        if self._settings.normalize_observations:
            self.policy.observation_normalizer.update(self._buffer.observations.flatten(0, 1))
        return collected.finished_episodes

    def report(self) -> dict[str, float]:
        """The statistics of the latest update (and the auxiliary objective's means since the previous report)."""
        record = {} if self._last is None else {key: float(value) for key, value in dataclasses.asdict(self._last).items()}
        return record if self._auxiliary is None else record | self._auxiliary.report()

    def extra_state(self) -> dict[str, Any]:
        state = {} if self._auxiliary is None else self._auxiliary.save_into({})
        return state if self._return_normalizer is None else state | {_RETURN_STATISTICS_KEY: self._return_normalizer.state_dict()}

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        if self._auxiliary is not None:
            self._auxiliary.load_from(state)
        if self._return_normalizer is not None:
            self._return_normalizer.load_state_dict(state[_RETURN_STATISTICS_KEY])
