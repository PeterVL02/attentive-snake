"""DQN as a ``Learner``: one iteration = one epsilon-greedy step of every env, then the gradient steps that are due."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from attentive_snake.env.types import EpisodeSummary
from attentive_snake.training.algorithm import LearnerSetup
from attentive_snake.training.dqn.collector import EpsilonGreedyCollector
from attentive_snake.training.dqn.epsilon import linear_epsilon
from attentive_snake.training.dqn.replay import ReplayBuffer
from attentive_snake.training.dqn.target import TargetNetwork
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer
from attentive_snake.training.dqn.update import dqn_gradient_step
from attentive_snake.training.schedule import crossed_multiple

EXPLORATION_STREAM, REPLAY_STREAM = 1, 2  # independent numpy streams derived from the seed


class DqnLearner:
    """Owns the optimizer, target network, replay buffer and collector; every schedule is a function of env steps.

    Per iteration (``num_envs`` env steps): epsilon from ``linear_epsilon``, one collector
    step into the buffer, then ``gradient_steps`` per multiple of ``train_frequency`` passed
    (once the buffer holds ``max(learning_starts, batch_size)`` transitions), then a target
    sync if a multiple of ``target_update_interval`` was passed. Random numbers come from
    numpy generators seeded from ``train.seed`` only, so periodic evaluation cannot change them.
    """

    def __init__(self, setup: LearnerSetup, replay: ReplayBuffer | None = None) -> None:
        train, settings, env = setup.train, setup.settings, setup.env
        self.policy = setup.policy
        self._train, self._settings, self._device = train, settings, setup.device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=train.learning_rate)
        self._target = TargetNetwork(self.policy)
        self._replay = replay if replay is not None else UniformReplayBuffer(
            settings.buffer_size, tuple(env.observation_shape), np.random.default_rng([train.seed, REPLAY_STREAM])
        )
        self._collector = EpsilonGreedyCollector(
            env,
            env.reset(seed=train.seed),
            np.random.default_rng([train.seed, EXPLORATION_STREAM]),
            setup.device,
            train.bootstrap_truncated,
        )
        self._steps_per_iteration = env.num_envs
        self._epsilon = settings.epsilon_start
        self._gradient_steps = 0
        self._window: list[tuple[float, float, float]] = []  # (loss, q_mean, mean |td error|) since the last report

    @property
    def steps_per_iteration(self) -> int:
        return self._steps_per_iteration

    def iterate(self, env_steps: int) -> tuple[EpisodeSummary, ...]:
        s = self._settings
        self._epsilon = linear_epsilon(s.epsilon_start, s.epsilon_end, s.epsilon_fraction, env_steps, self._train.total_env_steps)
        collected = self._collector.step(self.policy, self._epsilon)
        self._replay.add(collected.transitions)
        after = env_steps + self._steps_per_iteration
        rounds = after // s.train_frequency - env_steps // s.train_frequency
        if len(self._replay) >= max(s.learning_starts, s.batch_size):
            for _ in range(rounds * s.gradient_steps):
                self._gradient_step()
        if crossed_multiple(env_steps, after, s.target_update_interval):
            self._target.sync(self.policy)
        return collected.finished_episodes

    def report(self) -> dict[str, float]:
        """Means over the gradient steps since the previous report (none yet: only the counters)."""
        record = {"epsilon": self._epsilon, "replay_size": float(len(self._replay)), "gradient_steps": float(self._gradient_steps)}
        if self._window:
            loss, q_mean, td_error = np.mean(np.array(self._window), axis=0).tolist()
            record |= {"loss": loss, "q_mean": q_mean, "td_error": td_error}
        self._window = []
        return record

    def extra_state(self) -> dict[str, Any]:
        return {"target": self._target.state_dict()}

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        self._target.load_state_dict(state["target"])

    def _gradient_step(self) -> None:
        sample = self._replay.sample(self._settings.batch_size, self._device)
        stats = dqn_gradient_step(
            self.policy, self._target, self.optimizer, sample, self._train.gamma, self._settings.double, self._settings.max_grad_norm
        )
        self._replay.update_priorities(sample.indices, stats.td_errors.abs().cpu().numpy())
        self._gradient_steps += 1
        self._window.append((stats.loss, stats.q_mean, float(stats.td_errors.abs().mean())))
