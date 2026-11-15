"""From the console's screen to the picture the model sees: shrink to a square, scale to [0, 1].

Pure numpy (no OpenCV). Shrinking averages the screen pixels each output pixel covers (area
interpolation), written as two small matrices: ``picture = rows @ screen @ columns``.
"""

import numpy as np


def area_weights(source: int, target: int) -> np.ndarray:
    """``(target, source)`` weights averaging ``source`` pixels into ``target``: each row sums to 1 and
    weighs every source pixel by how much of it lies in that target pixel's span."""
    edges = np.linspace(0.0, source, target + 1)
    pixels = np.arange(source)
    overlap = np.minimum(edges[1:, None], pixels[None, :] + 1) - np.maximum(edges[:-1, None], pixels[None, :])
    weights = np.clip(overlap, 0.0, None)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


class ScreenShrinker:
    """``(H, W)`` or ``(H, W, 3)`` uint8 screens -> ``(1 or 3, size, size)`` float32 in [0, 1]."""

    def __init__(self, height: int, width: int, size: int) -> None:
        self._rows = area_weights(height, size) / np.float32(255.0)  # the scaling to [0, 1] rides along
        self._columns = np.ascontiguousarray(area_weights(width, size).T)

    def __call__(self, screen: np.ndarray) -> np.ndarray:
        planes = screen[None] if screen.ndim == 2 else screen.transpose(2, 0, 1)
        return np.clip(self._rows @ planes.astype(np.float32) @ self._columns, 0.0, 1.0)
