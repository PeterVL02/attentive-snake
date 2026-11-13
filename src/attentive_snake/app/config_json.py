"""Experiment configs as JSON: every field present (``None`` as ``null``), paths as strings."""

import dataclasses
from pathlib import Path
from typing import Any


def dataclass_to_json(obj: Any) -> dict[str, Any]:
    """Field values of a flat config dataclass, ``None`` kept, ``Path`` as a POSIX string."""
    return {
        field.name: _scalar(getattr(obj, field.name)) for field in dataclasses.fields(obj)
    }


def experiment_to_json(config: Any) -> dict[str, dict[str, Any]]:
    """``{section: {field: value}}`` for an ``ExperimentConfig``."""
    return {field.name: dataclass_to_json(getattr(config, field.name)) for field in dataclasses.fields(config)}


def _scalar(value: Any) -> Any:
    return value.as_posix() if isinstance(value, Path) else value
