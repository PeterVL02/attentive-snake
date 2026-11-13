"""Rolling stack of the last K frames for one environment."""

import numpy as np


class FrameHistory:
    """Owns a (K*F, H, W) buffer, oldest frame first, newest last.

    Frames before the episode start are zeros. ``reset`` and ``push`` return a
    fresh copy, so earlier observations are never mutated by later steps.
    """

    def __init__(self, length: int, frame_shape: tuple[int, int, int]):
        if length < 1:
            raise ValueError("history length must be at least 1.")
        channels, height, width = frame_shape
        self._channels = channels
        self._buffer = np.zeros((length * channels, height, width), dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._buffer.shape  # type: ignore[return-value]

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Start a new episode with ``frame`` as the only (newest) frame."""
        self._buffer.fill(0.0)
        self._buffer[-self._channels :] = frame
        return self._buffer.copy()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Drop the oldest frame, append ``frame`` as the newest."""
        c = self._channels
        self._buffer[:-c] = self._buffer[c:]  # numpy handles the overlap
        self._buffer[-c:] = frame
        return self._buffer.copy()
