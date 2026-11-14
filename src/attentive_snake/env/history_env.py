"""Any ``GameEnv`` with a history of frames (and optionally the actions that produced them)."""

import dataclasses
from typing import Any

import numpy as np

from attentive_snake.env.frame import ActionChannels
from attentive_snake.env.history import FrameHistory
from attentive_snake.env.types import EnvStep, GameEnv


class HistoryEnv:
    """Wraps a ``GameEnv`` into a ``SingleEnv``: observations are the last ``history_length`` frames.

    With ``history_actions`` each frame gets ``num_actions`` one-hot action channels (see
    ``ActionChannels``). Does not auto-reset; ``game`` is the wrapped environment (for
    game-specific inspection, e.g. Snake's ``game.state`` in tests).
    """

    def __init__(self, game: GameEnv, history_length: int = 1, history_actions: bool = False) -> None:
        self.game = game
        self.num_actions = game.num_actions
        self.history_length = history_length
        self.action_channels = game.num_actions if history_actions else 0
        self._actions = ActionChannels(tuple(game.observation_shape), self.action_channels)
        self.frame_channels = self._actions.shape[0]
        self._history = FrameHistory(history_length, self._actions.shape)
        self.observation_shape: tuple[int, ...] = self._history.shape

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._history.reset(self._actions.build(self.game.reset(seed)))

    def step(self, action: int) -> EnvStep:
        step = self.game.step(action)
        observation = self._history.push(self._actions.build(step.observation, action))
        return dataclasses.replace(step, observation=observation)

    def view(self) -> dict[str, Any]:
        return self.game.view()
