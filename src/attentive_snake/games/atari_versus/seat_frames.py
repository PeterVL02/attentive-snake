"""From the shared screen to each player's frame: the shrunk picture plus one plane that says which player it is.

Both players see the same screen, and the same network plays both, so the frame must say whose
turn at the joystick it is: a constant plane, 0 everywhere for the first player (seat 0) and 1
for the second. Constant planes are how action channels are given to image models as well
(``env.frame.ActionChannels``); the first convolution reads it like any other plane.
"""

import numpy as np

from attentive_snake.ale.screen import ScreenShrinker


class SeatFrames:
    """``screen -> (players, C + 1, size, size)`` float32: the picture (``C`` = 1 grayscale, 3 colour), then the seat plane."""

    def __init__(self, screen_shape: tuple[int, int], size: int, grayscale: bool, players: int) -> None:
        self._shrink = ScreenShrinker(*screen_shape, size)
        self._channels = 1 if grayscale else 3
        self._players = players
        self.shape: tuple[int, ...] = (self._channels + 1, size, size)  # one player's frame

    def __call__(self, screen: np.ndarray) -> np.ndarray:
        frames = np.empty((self._players, *self.shape), dtype=np.float32)
        frames[:, : self._channels] = self._shrink(screen)
        for seat in range(self._players):
            frames[seat, self._channels] = seat / (self._players - 1)
        return frames
