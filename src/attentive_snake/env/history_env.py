"""Any ``GameEnv`` with a history of frames (and optionally the actions that produced them)."""

import dataclasses
from typing import Any

import numpy as np

from attentive_snake.env.action_space import action_space_of
from attentive_snake.env.frame_stack import FrameStack
from attentive_snake.env.types import EnvStep, GameEnv


class HistoryEnv:
    """Wraps a ``GameEnv`` into a ``SingleEnv``: observations are the last ``history_length`` frames.

    With ``history_actions`` each frame gets ``num_actions`` one-hot action channels (see
    ``ActionChannels``); that needs discrete actions (a one-hot channel per action), so a game
    with continuous actions refuses it. ``step`` takes what a policy produced (an index, or a
    vector clipped here to the bounds: ``action_space.env_action``). Does not auto-reset;
    ``game`` is the wrapped environment (for game-specific inspection, e.g. Snake's
    ``game.state`` in tests).
    """

    def __init__(self, game: GameEnv, history_length: int = 1, history_actions: bool = False) -> None:
        self.game = game
        self.action_space = action_space_of(game)
        if history_actions and self.action_space.continuous:
            raise ValueError("history_actions needs discrete actions (one channel per action); this game's actions are continuous.")
        self.num_actions = game.num_actions
        self.history_length = history_length
        self.action_channels = game.num_actions if history_actions else 0
        self._frames = FrameStack(tuple(game.observation_shape), history_length, self.action_channels)
        self.frame_channels = self._frames.frame_channels
        self.observation_shape: tuple[int, ...] = self._frames.shape

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._frames.reset(self.game.reset(seed))

    def step(self, action: int | np.ndarray) -> EnvStep:
        action = self.action_space.env_action(action)
        step = self.game.step(action)
        return dataclasses.replace(step, observation=self._frames.push(step.observation, action))

    def view(self) -> dict[str, Any]:
        return self.game.view()
