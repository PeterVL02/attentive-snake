"""One displayable moment of a watched game."""

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WatchFrame:
    """Board, outcome and attention to show; ``to_dict`` gives plain JSON types.

    ``event`` is the ``StepEvent`` value of the step that produced the frame (``None`` right
    after a reset); ``done`` means that step ended the episode. ``heatmap`` (values in [0, 1],
    shape (height, width)) and ``frame_weights`` belong to the observation the *next* action
    is chosen from, and are ``None`` when not requested, not available or the episode is over.
    """

    width: int
    height: int
    snake: list[list[int]]  # [[x, y], ...] head first
    food: list[int] | None
    score: int
    steps: int
    event: str | None
    done: bool
    heatmap: list[list[float]] | None
    frame_weights: list[float] | None  # oldest frame first
    caption: str
    episode: int  # episodes completed so far

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
