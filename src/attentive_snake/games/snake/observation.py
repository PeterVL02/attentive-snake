"""Encodes a GameState as an image-like float32 array."""

import numpy as np

from attentive_snake.games.snake.types import GameState

HEAD, BODY, FOOD = 0, 1, 2


class ObservationEncoder:
    """GameState -> array of shape (3, height, width), indexed [channel, y, x].

    Channels: 0 head (1.0 at the head cell); 1 body: every snake cell,
    head included, holds ``(len - index) / len`` so the head is 1.0 and the
    tail is 1/len (how long until the cell is vacated); 2 food.
    """

    num_channels = 3

    def shape(self, width: int, height: int) -> tuple[int, int, int]:
        return (self.num_channels, height, width)

    def encode(self, state: GameState) -> np.ndarray:
        out = np.zeros(self.shape(state.width, state.height), dtype=np.float32)
        self.encode_into(state, out)
        return out

    def encode_into(self, state: GameState, out: np.ndarray) -> None:
        """Write the encoding into a preallocated array of the right shape."""
        out.fill(0.0)
        length = len(state.snake)
        for i, (x, y) in enumerate(state.snake):
            out[BODY, y, x] = (length - i) / length
        hx, hy = state.head
        out[HEAD, hy, hx] = 1.0
        if state.food is not None:
            fx, fy = state.food
            out[FOOD, fy, fx] = 1.0

    __call__ = encode
