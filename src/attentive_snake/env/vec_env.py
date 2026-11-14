"""Synchronous batch of single environments of any game."""

from collections.abc import Sequence

import numpy as np

from attentive_snake.env.episodes import EpisodeTracker
from attentive_snake.env.types import EpisodeSummary, SingleEnv, VecEnvStep


def env_seed(seed: int | None, index: int) -> int | None:
    """The seed of sub-environment ``index`` derived from one experiment seed (``None`` stays ``None``)."""
    if seed is None:
        return None
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


class SyncVecEnv:
    """N ``SingleEnv``s stepped in a Python loop; finished envs are reset automatically.

    All envs must agree on ``observation_shape`` and ``num_actions``. ``reset(seed)``
    gives env ``i`` the seed ``env_seed(seed, i)``.
    """

    def __init__(self, envs: Sequence[SingleEnv]) -> None:
        if not envs:
            raise ValueError("a vector env needs at least one environment.")
        first = envs[0]
        for env in envs[1:]:
            if tuple(env.observation_shape) != tuple(first.observation_shape) or env.num_actions != first.num_actions:
                raise ValueError("all environments of a vector env must have the same observation shape and actions.")
        self.envs: tuple[SingleEnv, ...] = tuple(envs)
        self.num_envs = len(self.envs)
        self.num_actions = first.num_actions
        self.observation_shape: tuple[int, ...] = tuple(first.observation_shape)
        self.history_length = first.history_length
        self.frame_channels = first.frame_channels
        self.action_channels = first.action_channels
        self._tracker = EpisodeTracker(self.num_envs)
        self._obs = np.zeros((self.num_envs, *self.observation_shape), dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._tracker.reset()
        for i, env in enumerate(self.envs):
            self._obs[i] = env.reset(env_seed(seed, i))
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> VecEnvStep:
        n = self.num_envs
        rewards = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        finished: list[EpisodeSummary] = []
        for i, env in enumerate(self.envs):
            result = env.step(int(actions[i]))
            rewards[i] = result.reward
            terminated[i] = result.terminated
            truncated[i] = result.truncated
            summary = self._tracker.update(i, result)
            if summary is not None:
                finished.append(summary)
                self._obs[i] = env.reset()
            else:
                self._obs[i] = result.observation
        return VecEnvStep(self._obs.copy(), rewards, terminated, truncated, tuple(finished))
