"""One displayable moment of a watched game."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WatchFrame:
    """The game's view payload plus what is the same for every game; ``to_dict`` gives plain JSON types.

    ``view`` is the payload of the game's ``view_kind`` (see ``attentive_snake.views``);
    ``to_dict`` merges it into the top level, next to the generic keys below.
    ``score`` is the game's headline number for the episode so far, ``steps`` the steps
    taken in it. ``event`` is the game's word for what the step that produced the frame
    did (``None`` right after a reset); ``done`` means that step ended the episode.
    ``heatmap`` (values in [0, 1], the observation's (height, width)) and
    ``frame_weights`` belong to the observation the *next* action is chosen from, and are
    ``None`` when not requested, not available or the episode is over.
    """

    view_kind: str
    view: dict[str, Any]
    score: float
    steps: int
    event: str | None
    done: bool
    heatmap: list[list[float]] | None
    frame_weights: list[float] | None  # oldest frame first
    caption: str
    episode: int  # episodes completed so far

    def to_dict(self) -> dict[str, Any]:
        generic = {
            "view_kind": self.view_kind,
            "score": self.score,
            "steps": self.steps,
            "event": self.event,
            "done": self.done,
            "heatmap": self.heatmap,
            "frame_weights": self.frame_weights,
            "caption": self.caption,
            "episode": self.episode,
        }
        clashes = set(self.view) & set(generic)
        if clashes:
            raise ValueError(f"view payload keys {sorted(clashes)} clash with the generic watch-frame keys.")
        return {**self.view, **generic}
