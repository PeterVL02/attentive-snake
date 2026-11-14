"""Builds one observation frame: board channels, optionally followed by action planes."""

import numpy as np

from attentive_snake.games.snake.observation import ObservationEncoder
from attentive_snake.games.snake.types import Action, GameState

NUM_ACTION_PLANES = len(Action)


class FrameBuilder:
    """GameState (+ the action just taken) -> float32 array (F, height, width).

    Channels: the encoder's board channels, then (if ``with_actions``) one
    constant plane per ``Action`` value, 1.0 on the plane of the chosen action.
    ``action=None`` (the reset frame) leaves all action planes zero.
    """

    def __init__(self, encoder: ObservationEncoder, with_actions: bool = False):
        self._encoder = encoder
        self._with_actions = with_actions
        self._board = encoder.num_channels
        self.channels = self._board + (NUM_ACTION_PLANES if with_actions else 0)

    def shape(self, width: int, height: int) -> tuple[int, int, int]:
        return (self.channels, height, width)

    def build(self, state: GameState, action: int | None = None) -> np.ndarray:
        if not self._with_actions:
            return self._encoder.encode(state)
        frame = np.zeros(self.shape(state.width, state.height), dtype=np.float32)
        self._encoder.encode_into(state, frame[: self._board])
        if action is not None:
            frame[self._board + int(action)] = 1.0
        return frame
