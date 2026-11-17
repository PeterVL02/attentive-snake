"""A ``"board"`` payload made of named, coloured layers of cells (a grid game with several kinds of objects).

::

    {"width": W, "height": H, "snake": [], "foods": [],
     "layers": [{"name": "ball", "color": "#e8a33d", "cells": [[x, y], ...]}, ...]}

Each layer is one kind of object (MinAtar: one observation channel), drawn in its colour in
list order (later layers over earlier ones) and named in a legend. ``snake`` and ``foods`` stay
(empty), so every display of the board kind can draw it; see ``views.board``.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from attentive_snake.views.board import Cell, board_payload


def board_layer(name: str, color: str, cells: Iterable[Cell]) -> dict[str, Any]:
    """One layer: its legend name, a CSS colour ``#rrggbb`` and its cells ``[x, y]``."""
    return {"name": name, "color": color, "cells": [[int(x), int(y)] for x, y in cells]}


def layered_board_payload(width: int, height: int, layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The board payload with no player body or targets, and ``layers`` (from ``board_layer``)."""
    return {**board_payload(width, height, [], []), "layers": list(layers)}
