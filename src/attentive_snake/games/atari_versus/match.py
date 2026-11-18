"""A two-player Atari match as a ``MatchEnv`` (``env.types``): both players' frames, rewards and events, for self-play.

One step presses one action per player (``Console.hold``). Each player's frame is the shared
screen plus its seat plane (``SeatFrames``), its reward its own points (``reward.versus_reward``),
its score the sum of its points (zero-sum titles: its margin over the other player). Both end on
the same step: ``terminated`` when the title ends the match, ``truncated`` at
``max_episode_steps``. ``view()`` is the console's full colour screen.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from attentive_snake.ale.actions import labels_of_ids
from attentive_snake.env.types import EnvStep
from attentive_snake.games.atari_versus.config import VersusRules
from attentive_snake.games.atari_versus.console import PLAYERS, Console
from attentive_snake.games.atari_versus.outcome import seat_event
from attentive_snake.games.atari_versus.reward import VersusReward, versus_reward
from attentive_snake.games.atari_versus.seat_frames import SeatFrames
from attentive_snake.views.image import image_payload


class VersusMatch:
    """One match of ``rules.title``; ``reset`` gives ``(2, *observation_shape)``, ``step`` one ``EnvStep`` per seat."""

    num_seats = PLAYERS

    def __init__(self, rules: VersusRules, reward: VersusReward, seed: int | None = None) -> None:
        self._console = Console(rules)
        self._frames = SeatFrames(self._console.screen_shape, rules.screen_size, rules.grayscale, PLAYERS)
        self._rules, self._reward = rules, reward
        self._seed = seed  # used by the first reset that is given no seed
        self.observation_shape = self._frames.shape
        self.num_actions = len(self._console.action_ids)
        self.action_labels = labels_of_ids(self._console.action_ids)
        self._scores, self._steps = np.zeros(PLAYERS), 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        seed, self._seed = (seed if seed is not None else self._seed), None
        self._console.start(seed)
        self._scores, self._steps = np.zeros(PLAYERS), 0
        return self._frames(self._console.screen())

    def step(self, actions: Sequence[int]) -> tuple[EnvStep, ...]:
        points, game_over = self._console.hold([self._console.action_ids[int(action)] for action in actions])
        self._scores += points
        self._steps += 1
        timed_out = not game_over and self._steps >= self._rules.max_episode_steps
        frames = self._frames(self._console.screen())
        return tuple(
            EnvStep(
                observation=frames[seat],
                reward=versus_reward(self._reward, points[seat]),
                terminated=game_over,
                truncated=timed_out,
                event=seat_event(points[seat], self._scores[seat], game_over, timed_out),
                score=float(self._scores[seat]),
            )
            for seat in range(PLAYERS)
        )

    def view(self) -> dict[str, Any]:
        return image_payload(self._console.rgb())
