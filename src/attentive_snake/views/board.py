"""The ``"board"`` view payload: a grid with a player body and one target cell.

::

    {"width": W, "height": H,                  # grid size in cells
     "snake": [[x, y], ...],                   # player body cells, head first (may be empty)
     "food": [x, y] or null}                   # the target cell, or none

Coordinates: x grows to the right, y downwards, (0, 0) is the top-left cell. The app's
``board.js`` and ``views.board_window`` draw exactly these keys (body green, head brighter,
target red); a heatmap from the watch frame, shape (H, W), is drawn under them.
"""

from collections.abc import Iterable, Sequence
from typing import Any

type Cell = Sequence[int]


def board_payload(width: int, height: int, body: Iterable[Cell], target: Cell | None) -> dict[str, Any]:
    """The board payload; ``body`` is head first."""
    return {
        "width": int(width),
        "height": int(height),
        "snake": [[int(x), int(y)] for x, y in body],
        "food": None if target is None else [int(target[0]), int(target[1])],
    }
