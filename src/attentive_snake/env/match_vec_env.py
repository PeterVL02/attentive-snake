"""Self-play: matches of several seats as one batch of environments, one slot per seat."""

from collections.abc import Sequence

import numpy as np

from attentive_snake.env.action_space import action_space_of
from attentive_snake.env.episodes import EpisodeTracker
from attentive_snake.env.frame_stack import FrameStack
from attentive_snake.env.types import EpisodeSummary, MatchEnv, VecEnvStep
from attentive_snake.env.vec_env import env_seed


class MatchVecEnv:
    """M ``MatchEnv``s of S seats as one ``VecEnv`` of M * S slots: slot ``m * S + s`` is seat ``s`` of match ``m``.

    Each slot is one seat's trajectory, as a ``SyncVecEnv`` slot is one game's: its own history of
    frames (``FrameStack``, with the seat's own actions as action channels), reward and episode
    summary. One ``step`` takes an action for every slot and steps every match once with the
    actions of its seats. A match ends for all its seats on the same step (``ValueError``
    otherwise): their last observations go in ``final_observations`` and the match is reset, so
    every slot of it starts its next episode together. ``reset(seed)`` gives match ``m`` the seed
    ``env_seed(seed, m)``. Discrete actions only.
    """

    def __init__(self, matches: Sequence[MatchEnv], history_length: int = 1, history_actions: bool = False) -> None:
        if not matches:
            raise ValueError("a vector env needs at least one match.")
        first = matches[0]
        for match in matches[1:]:
            same = (match.num_seats, match.num_actions, tuple(match.observation_shape)) == (first.num_seats, first.num_actions, tuple(first.observation_shape))
            if not same:
                raise ValueError("all matches of a vector env must have the same seats, observation shape and actions.")
        self.action_space = action_space_of(first)
        if self.action_space.continuous:
            raise ValueError("self-play matches need discrete actions.")
        self.matches: tuple[MatchEnv, ...] = tuple(matches)
        self.num_seats = first.num_seats
        self.num_envs = len(self.matches) * self.num_seats
        self.num_actions = first.num_actions
        self.history_length = history_length
        self.action_channels = first.num_actions if history_actions else 0
        self._stacks = [FrameStack(tuple(first.observation_shape), history_length, self.action_channels) for _ in range(self.num_envs)]
        self.frame_channels = self._stacks[0].frame_channels
        self.observation_shape: tuple[int, ...] = self._stacks[0].shape
        self._tracker = EpisodeTracker(self.num_envs)
        self._obs = np.zeros((self.num_envs, *self.observation_shape), dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._tracker.reset()
        for index, match in enumerate(self.matches):
            self._start(index, match.reset(env_seed(seed, index)))
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> VecEnvStep:
        n = self.num_envs
        rewards = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        finished: list[EpisodeSummary] = []
        final_observations: dict[int, np.ndarray] = {}
        for index, match in enumerate(self.matches):
            slots = self._slots(index)
            seat_actions = [self.action_space.env_action(actions[slot]) for slot in slots]
            results = match.step(seat_actions)
            ends = {(result.terminated, result.truncated) for result in results}
            if len(ends) != 1:
                raise ValueError("all seats of a match must end on the same step, in the same way.")
            for slot, action, result in zip(slots, seat_actions, results, strict=True):
                self._obs[slot] = self._stacks[slot].push(result.observation, action)
                rewards[slot], terminated[slot], truncated[slot] = result.reward, result.terminated, result.truncated
                summary = self._tracker.update(slot, result)
                if summary is not None:
                    finished.append(summary)
                    final_observations[slot] = self._obs[slot].copy()
            if results[0].terminated or results[0].truncated:
                self._start(index, match.reset())
        return VecEnvStep(self._obs.copy(), rewards, terminated, truncated, tuple(finished), final_observations)

    def _slots(self, index: int) -> range:
        return range(index * self.num_seats, (index + 1) * self.num_seats)

    def _start(self, index: int, frames: np.ndarray) -> None:
        """Start the episodes of match ``index``'s slots from its first frames (one per seat)."""
        for seat, slot in enumerate(self._slots(index)):
            self._obs[slot] = self._stacks[slot].reset(frames[seat])
