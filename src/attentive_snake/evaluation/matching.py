"""Whether a new evaluation reproduces one recorded earlier: same checkpoint, same games, same numbers."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from attentive_snake.evaluation.record import Evaluation


@dataclass(frozen=True)
class Reproduction:
    """The latest recorded evaluation of the same checkpoint with the same games, and whether the numbers agree.

    ``same_device`` is false when one ran on another device (e.g. GPU vs CPU); float
    results may then differ in the last bits, and so, rarely, may a game.
    """

    recorded: Evaluation
    identical: bool
    same_device: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.recorded.timestamp,
            "source": self.recorded.source,
            "identical": self.identical,
            "same_device": self.same_device,
        }


def find_reproduction(recorded: Sequence[Evaluation], new: Evaluation) -> Reproduction | None:
    """Compare ``new`` with the latest of ``recorded`` that evaluated the same checkpoint the same way; ``None`` if there is none."""
    candidates = [
        old for old in recorded
        if old is not new and old.checkpoint == new.checkpoint and old.settings.same_games(new.settings)
    ]
    if not candidates:
        return None
    old = candidates[-1]
    identical = old.statistics == new.statistics and old.outcomes == new.outcomes
    return Reproduction(old, identical, old.settings.device == new.settings.device)
