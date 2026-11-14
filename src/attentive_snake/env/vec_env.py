"""Synchronous batch of snake environments."""

import numpy as np

from attentive_snake.games.snake.reward import RewardConfig, RewardTerm
from attentive_snake.games.snake.env import SnakeEnv
from attentive_snake.env.types import EpisodeSummary, VecEnvStep
from attentive_snake.games.snake.config import GameConfig
from attentive_snake.games.snake.types import GameState, StepEvent


class EpisodeTracker:
    """Accumulates per-env return and emits an EpisodeSummary when an episode ends."""

    def __init__(self, num_envs: int):
        self._returns = [0.0] * num_envs

    def reset(self) -> None:
        self._returns = [0.0] * len(self._returns)

    def update(
        self, env_index: int, reward: float, finished: bool, state: GameState, event: StepEvent
    ) -> EpisodeSummary | None:
        """Add a step's reward; return a summary (and clear the return) if finished."""
        self._returns[env_index] += reward
        if not finished:
            return None
        total, self._returns[env_index] = self._returns[env_index], 0.0
        return EpisodeSummary(
            env_index=env_index,
            score=state.score,
            length=state.steps,
            total_reward=total,
            final_event=event,
        )


class SyncVecSnakeEnv:
    """N SnakeEnvs stepped in a Python loop; finished envs are reset automatically."""

    def __init__(
        self,
        num_envs: int,
        game_config: GameConfig,
        reward_config: RewardConfig | RewardTerm | None = None,
        seed: int | None = None,
        history_length: int = 1,
        history_actions: bool = False,
    ):
        self.num_envs = num_envs
        self.num_actions = SnakeEnv.num_actions
        self._envs = [
            SnakeEnv(
                game_config,
                reward_config,
                seed=self._derive(seed, i),
                history_length=history_length,
                history_actions=history_actions,
            )
            for i in range(num_envs)
        ]
        self.observation_shape = self._envs[0].observation_shape
        self.history_length = history_length
        self.frame_channels = self._envs[0].frame_channels
        self._tracker = EpisodeTracker(num_envs)
        shape = (num_envs, *self.observation_shape)
        self._obs = np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _derive(seed: int | None, index: int) -> int | None:
        if seed is None:
            return None
        return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._tracker.reset()
        for i, env in enumerate(self._envs):
            self._obs[i] = env.reset(self._derive(seed, i))
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> VecEnvStep:
        n = self.num_envs
        rewards = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        finished: list[EpisodeSummary] = []
        for i, env in enumerate(self._envs):
            result = env.step(int(actions[i]))
            rewards[i] = result.reward
            terminated[i] = result.terminated
            truncated[i] = result.truncated
            done = result.terminated or result.truncated
            summary = self._tracker.update(i, result.reward, done, env.state, result.event)
            if summary is not None:
                finished.append(summary)
                self._obs[i] = env.reset()
            else:
                self._obs[i] = result.observation
        return VecEnvStep(self._obs.copy(), rewards, terminated, truncated, tuple(finished))

    def states(self) -> list[GameState]:
        """Current game state of every sub-env."""
        return [env.state for env in self._envs]
