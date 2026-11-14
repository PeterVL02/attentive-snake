"""``Evaluation``: one evaluation of one set of weights, as stored in a run's evaluations.jsonl.

JSON (one line per evaluation)::

    {"timestamp": "2026-11-14T10:00:00+00:00", "source": "training", "env_steps": 819200,
     "checkpoint": "checkpoint_0000819200.pt",
     "settings": {"games": 100, "seed": 1000, "max_episode_steps": 10000, "parallel_games": 32,
                  "device": "cpu", "policy": "greedy"},
     "statistics": {"win_rate": 0.02, "mean_steps_to_win": 512.0, ..., "mean_score": 21.4, ...},
     "outcomes": {"hit_self": 61, "hit_wall": 37, "won": 2}, "seconds": 3.1}

``source`` is ``"training"`` (made by the run while it trained) or ``"manual"`` (made
afterwards by ``scripts/evaluate.py --save`` or the app). ``checkpoint`` is the file name of
the evaluated weights in the run directory (``null`` only for training without checkpoints).
A statistic that could not be computed is ``null``.
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Literal

from attentive_snake.episode_metrics import Statistics
from attentive_snake.evaluation.settings import EvalSettings

type Source = Literal["training", "manual"]
SOURCES = ("training", "manual")


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Evaluation:
    source: Source
    env_steps: int
    checkpoint: str | None
    settings: EvalSettings
    statistics: Statistics
    outcomes: dict[str, int]
    seconds: float
    timestamp: str = field(default_factory=now)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}, not {self.source!r}.")

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "env_steps": self.env_steps,
            "checkpoint": self.checkpoint,
            "settings": self.settings.to_json(),
            "statistics": dict(self.statistics),
            "outcomes": dict(self.outcomes),
            "seconds": self.seconds,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Evaluation":
        """``KeyError``/``TypeError``/``ValueError`` for anything that is not an evaluation record."""
        return cls(
            source=data["source"],
            env_steps=int(data["env_steps"]),
            checkpoint=data["checkpoint"],
            settings=EvalSettings.from_json(data["settings"]),
            statistics=dict(data["statistics"]),
            outcomes={str(key): int(value) for key, value in data["outcomes"].items()},
            seconds=float(data["seconds"]),
            timestamp=str(data["timestamp"]),
        )
