"""Training metrics: aggregating episode statistics and sending metric records to a sink."""

import json
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from attentive_snake.env.types import EpisodeSummary


class MetricsLogger(Protocol):
    """Anything that can receive one record of scalar metrics. Plug in your own."""

    def log(self, env_steps: int, metrics: Mapping[str, float]) -> None: ...


class ConsoleLogger:
    """Prints each record as a single line."""

    def log(self, env_steps: int, metrics: Mapping[str, float]) -> None:
        fields = " ".join(f"{name}={_format(value)}" for name, value in metrics.items())
        print(f"steps={env_steps} {fields}", flush=True)


class TeeLogger:
    """Forwards every record to each of the given loggers."""

    def __init__(self, *loggers: MetricsLogger) -> None:
        self._loggers = loggers

    def log(self, env_steps: int, metrics: Mapping[str, float]) -> None:
        for logger in self._loggers:
            logger.log(env_steps, metrics)


class JsonlLogger:
    """Appends each record as one JSON line ``{"env_steps": ..., **metrics}``; flushed per line."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def log(self, env_steps: int, metrics: Mapping[str, float]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {"env_steps": int(env_steps), **{k: v for k, v in metrics.items() if k != "env_steps"}}
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """All records of a metrics file; ``[]`` if it is missing, a truncated or corrupt line is skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


class NullLogger:
    """Discards everything."""

    def log(self, env_steps: int, metrics: Mapping[str, float]) -> None:
        pass


class EpisodeStatsWindow:
    """Running means over the most recent finished episodes."""

    def __init__(self, window: int = 100) -> None:
        self._episodes: deque[EpisodeSummary] = deque(maxlen=window)
        self._total_finished = 0

    def add(self, episodes: Iterable[EpisodeSummary]) -> None:
        for episode in episodes:
            self._episodes.append(episode)
            self._total_finished += 1

    def summary(self) -> dict[str, float]:
        """Means over the window; empty if no episode has finished yet."""
        if not self._episodes:
            return {}
        count = len(self._episodes)
        return {
            "episodes": float(self._total_finished),
            "mean_score": sum(e.score for e in self._episodes) / count,
            "mean_length": sum(e.length for e in self._episodes) / count,
            "mean_return": sum(e.total_reward for e in self._episodes) / count,
        }


def _format(value: float) -> str:
    if float(value).is_integer() and abs(value) < 1e12:
        return str(int(value))
    return f"{value:.4g}"
