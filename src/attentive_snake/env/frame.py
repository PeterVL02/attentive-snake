"""Appending one-hot action channels to a frame of any rank."""

import numpy as np


class ActionChannels:
    """A frame ``(F, *rest)`` -> ``(F + A, *rest)``: the frame, then one constant channel per action.

    The channel of the chosen action is 1.0 everywhere, the others 0; ``action=None``
    (the first frame of an episode) leaves all of them zero. With ``num_actions = 0``
    the frame is returned unchanged. ``rest`` is ``(H, W)`` for grids and images and
    ``()`` for vectors, where the channels are simply ``A`` extra numbers.
    """

    def __init__(self, frame_shape: tuple[int, ...], num_actions: int) -> None:
        self._base = frame_shape[0]
        self._num_actions = num_actions
        self.shape: tuple[int, ...] = (frame_shape[0] + num_actions, *frame_shape[1:])

    def build(self, frame: np.ndarray, action: int | None = None) -> np.ndarray:
        if not self._num_actions:
            return frame
        out = np.zeros(self.shape, dtype=np.float32)
        out[: self._base] = frame
        if action is not None:
            out[self._base + int(action)] = 1.0
        return out
