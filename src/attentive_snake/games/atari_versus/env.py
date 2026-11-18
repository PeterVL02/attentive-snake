"""The model's player against the random one, as a ``GameEnv``: what evaluation and watching play.

Self-play returns hover around zero whatever the model has learned, so progress is measured
against a fixed opponent (``opponent.RandomOpponent``). The model's seat alternates: a seeded
game puts it in seat ``seed % 2`` (an evaluation's games, ``env_seed(seed, k)``, fall about half
in each), and each unseeded reset in the other seat than the game before (watching). Its frame,
reward, events and score are its seat's of the match (``match.VersusMatch``); ``view()`` adds a
``note`` naming the player it controls (``views``).
"""

from typing import Any

import numpy as np

from attentive_snake.env.types import EnvStep
from attentive_snake.games.atari_versus.config import VersusRules
from attentive_snake.games.atari_versus.match import VersusMatch
from attentive_snake.games.atari_versus.opponent import RandomOpponent
from attentive_snake.games.atari_versus.reward import VersusReward
from attentive_snake.games.atari_versus.titles import get_title


def model_seat(seed: int | None, previous: int) -> int:
    """The model's seat in a game started with ``seed`` (``previous``: its seat in the game before)."""
    return int(seed) % 2 if seed is not None else 1 - previous


class VersusEnv:
    """One match of ``rules.title``: the model in one seat, ``RandomOpponent`` in the other."""

    def __init__(self, rules: VersusRules, reward: VersusReward, seed: int | None = None) -> None:
        self._match = VersusMatch(rules, reward)
        self._opponent = RandomOpponent(self._match.num_actions)
        self._players = get_title(rules.title).players
        self._seed = seed  # used by the first reset that is given no seed
        self._seat = 1  # so that the first unseeded game puts the model in seat 0
        self.observation_shape = self._match.observation_shape
        self.num_actions = self._match.num_actions
        self.action_labels = self._match.action_labels

    @property
    def seat(self) -> int:
        """The model's seat in the current game."""
        return self._seat

    def reset(self, seed: int | None = None) -> np.ndarray:
        seed, self._seed = (seed if seed is not None else self._seed), None
        self._seat = model_seat(seed, self._seat)
        self._opponent.reset(seed)
        return self._match.reset(seed)[self._seat]

    def step(self, action: int | np.ndarray) -> EnvStep:
        actions = [self._opponent.act()] * self._match.num_seats
        actions[self._seat] = int(action)
        return self._match.step(actions)[self._seat]

    def view(self) -> dict[str, Any]:
        mine, theirs = self._players[self._seat], self._players[1 - self._seat]
        return {**self._match.view(), "note": f"The model plays the {mine}; the {theirs} presses random buttons."}
