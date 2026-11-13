"""Reading and writing an ``ExperimentConfig`` as a TOML file.

One TOML table per ``ExperimentConfig`` field: ``[game] [reward] [model]
[train] [env]``. Missing sections and keys take the dataclass defaults;
unknown sections or keys are errors.
"""

import dataclasses
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.experiment import ExperimentConfig
from attentive_snake.config.toml_writer import Scalar, format_tables

_SECTIONS = {field.name: field for field in dataclasses.fields(ExperimentConfig)}


def experiment_from_tables(tables: Mapping[str, Any]) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from parsed TOML (``{section: {key: value}}``)."""
    sections = {}
    for name, table in tables.items():
        if name not in _SECTIONS:
            raise ValueError(f"unknown section [{name}]; valid sections: {', '.join(_SECTIONS)}.")
        if not isinstance(table, Mapping):
            raise TypeError(f"[{name}] must be a table, got {type(table).__name__}.")
        sections[name] = dataclass_from_table(_SECTIONS[name].type, table, section=name)
    return ExperimentConfig(**sections)


def experiment_to_tables(config: ExperimentConfig) -> dict[str, dict[str, Scalar]]:
    """``{section: {key: value}}`` for every section, ``None`` values omitted."""
    return {name: dataclass_to_table(getattr(config, name)) for name in _SECTIONS}


def load_experiment(path: Path) -> ExperimentConfig:
    with Path(path).open("rb") as file:
        tables = tomllib.load(file)
    try:
        return experiment_from_tables(tables)
    except (ValueError, TypeError) as error:
        raise type(error)(f"{path}: {error}") from error


def save_experiment(config: ExperimentConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_tables(experiment_to_tables(config)), encoding="utf-8")
