"""Single RL environment: game + reward + observation."""

import numpy as np

from attentive_snake.env.frame import FrameBuilder
from attentive_snake.env.history import FrameHistory
from attentive_snake.env.observation import ObservationEncoder
from attentive_snake.env.reward import RewardConfig, RewardTerm, resolve_reward
from attentive_snake.env.types import EnvStep
from attentive_snake.game.config import GameConfig
from attentive_snake.game.engine import SnakeGame
from attentive_snake.game.types import Action, GameState, StepEvent


class SnakeEnv:
    """One snake environment. Does not auto-reset; call ``reset`` after an episode ends."""

    num_actions = len(Action)

    def __init__(
        self,
        game_config: GameConfig,
        reward_config: RewardConfig | RewardTerm | None = None,
        encoder: ObservationEncoder | None = None,
        seed: int | None = None,
        history_length: int = 1,
        history_actions: bool = False,
    ):
        self._game = SnakeGame(game_config, seed)
        self._reward = resolve_reward(reward_config)
        self._encoder = encoder or ObservationEncoder()
        self._frames = FrameBuilder(self._encoder, history_actions)
        self.history_length = history_length
        self.frame_channels = self._frames.channels
        self._history = FrameHistory(
            history_length, self._frames.shape(game_config.width, game_config.height)
        )
        self.observation_shape = self._history.shape

    @property
    def state(self) -> GameState:
        return self._game.state

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._history.reset(self._frames.build(self._game.reset(seed)))

    def step(self, action: int) -> EnvStep:
        previous = self._game.state
        action = Action(int(action))
        result = self._game.step(action)
        event: StepEvent = result.event
        return EnvStep(
            observation=self._history.push(self._frames.build(result.state, action)),
            reward=self._reward(previous, result),
            terminated=event.is_terminal,
            truncated=event.is_truncation,
            event=event,
        )
