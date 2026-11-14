"""Experiment configs as JSON: every field present (``None`` as ``null``), paths as strings."""

import dataclasses
from pathlib import Path
from typing import Any

from attentive_snake.config.experiment import GAME_NAME_KEY, ExperimentConfig


def dataclass_to_json(obj: Any) -> dict[str, Any]:
    """Field values of a flat config dataclass, ``None`` kept, ``Path`` as a POSIX string."""
    return {
        field.name: _scalar(getattr(obj, field.name)) for field in dataclasses.fields(obj)
    }


def experiment_to_json(config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """``{section: {field: value}}`` for an ``ExperimentConfig``; ``game`` starts with ``name`` (the game)."""
    tables = {section: dataclass_to_json(value) for section, value in config.sections().items()}
    tables["game"] = {GAME_NAME_KEY: config.game_name, **tables["game"]}
    return tables


def _scalar(value: Any) -> Any:
    return value.as_posix() if isinstance(value, Path) else value
