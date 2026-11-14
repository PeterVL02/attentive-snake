"""Layout of a run directory, and finding runs.

A run directory holds everything about one training run::

    runs/<name>/
        config.toml                    the full ExperimentConfig (written at start)
        model.html                     interactive model diagram
        metrics.jsonl                  one JSON line per logged training update
        run.json                       history: start / resume / fork events with descriptions
        checkpoint_<env_steps>.pt      policy + optimizer + env-step counter

Checkpoints are found with ``attentive_snake.training.latest_checkpoint``; ``run.json``
is read and written by ``attentive_snake.run_metadata``.
"""

import re
from pathlib import Path

from attentive_snake.training.checkpoint import checkpoint_steps, list_checkpoints

CONFIG_FILENAME = "config.toml"
DIAGRAM_FILENAME = "model.html"
METRICS_FILENAME = "metrics.jsonl"
METADATA_FILENAME = "run.json"

_RUN_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")

__all__ = [
    "checkpoint_steps",
    "config_path",
    "diagram_path",
    "list_checkpoints",
    "list_runs",
    "metadata_path",
    "metrics_path",
    "validate_run_name",
]


def config_path(run_dir: Path) -> Path:
    return Path(run_dir) / CONFIG_FILENAME


def diagram_path(run_dir: Path) -> Path:
    return Path(run_dir) / DIAGRAM_FILENAME


def metrics_path(run_dir: Path) -> Path:
    return Path(run_dir) / METRICS_FILENAME


def metadata_path(run_dir: Path) -> Path:
    return Path(run_dir) / METADATA_FILENAME


def list_runs(root: Path) -> list[str]:
    """Names of the run directories (those with a config.toml) under ``root``, most recently modified first."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs = [path for path in root.iterdir() if path.is_dir() and config_path(path).is_file()]
    runs.sort(key=_last_modified, reverse=True)
    return [path.name for path in runs]


def _last_modified(run_dir: Path) -> float:
    """Newest of the directory (files added) and its metrics file (appended to in place)."""
    times = [run_dir.stat().st_mtime]
    if metrics_path(run_dir).is_file():
        times.append(metrics_path(run_dir).stat().st_mtime)
    return max(times)


def validate_run_name(name: str) -> str:
    """``name`` unchanged if it is only ``[A-Za-z0-9._-]`` and does not start with a dot, else ``ValueError``."""
    if not _RUN_NAME.fullmatch(name):
        raise ValueError(
            f"invalid name {name!r}: use letters, digits, '.', '_' and '-' only, not starting with a dot."
        )
    return name
