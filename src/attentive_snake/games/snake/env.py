"""Snake as a ``GameEnv``: game + reward + observation encoder, one frame per observation."""

from typing import Any

import numpy as np

from attentive_snake.env.types import EnvStep
from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.engine import SnakeGame
from attentive_snake.games.snake.observation import ObservationEncoder
from attentive_snake.games.snake.reward import RewardTerm, SnakeRewardConfig, resolve_reward
from attentive_snake.games.snake.types import Action, GameState
from attentive_snake.games.snake.view import board_view


class SnakeEnv:
    """One snake environment with ``(3, height, width)`` observations. Does not auto-reset.

    ``event`` of a step is the ``StepEvent`` value (``"moved"``, ``"ate_food"``,
    ``"hit_wall"``, ``"hit_self"``, ``"won"``, ``"starved"``); ``score`` is the food eaten.
    """

    num_actions = len(Action)

    def __init__(
        self,
        config: SnakeConfig,
        reward: SnakeRewardConfig | RewardTerm | None = None,
        seed: int | None = None,
        encoder: ObservationEncoder | None = None,
    ) -> None:
        self._game = SnakeGame(config, seed)
        self._reward = resolve_reward(reward)
        self._encoder = encoder or ObservationEncoder()
        self.observation_shape = self._encoder.shape(config.width, config.height)

    @property
    def state(self) -> GameState:
        return self._game.state

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._encoder.encode(self._game.reset(seed))

    def step(self, action: int) -> EnvStep:
        previous = self._game.state
        result = self._game.step(Action(int(action)))
        return EnvStep(
            observation=self._encoder.encode(result.state),
            reward=self._reward(previous, result),
            terminated=result.event.is_terminal,
            truncated=result.event.is_truncation,
            event=result.event.value,
            score=float(result.state.score),
        )

    def view(self) -> dict[str, Any]:
        return board_view(self._game.state)
