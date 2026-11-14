"""Reading and writing an ``ExperimentConfig`` as a TOML file.

One TOML table per section: ``[game] [reward] [model] [train] [env] [evaluation]``. ``[game] name``
selects the game (missing: ``"snake"``, so configs written before there were several
games still load); the other ``[game]`` keys and all of ``[reward]`` are read with that
game's dataclasses. Missing sections and keys take the dataclass defaults (so configs written before
``[evaluation]`` existed load with evaluation off); unknown
sections or keys are errors.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.experiment import GAME_NAME_KEY, SECTIONS, ExperimentConfig, section_types
from attentive_snake.config.toml_writer import Scalar, format_tables
from attentive_snake.games.registry import DEFAULT_GAME


def game_name_of(tables: Mapping[str, Any]) -> str:
    """The game an experiment's tables are for (``[game] name``, default ``"snake"``)."""
    game_table = tables.get("game", {})
    name = game_table.get(GAME_NAME_KEY, DEFAULT_GAME) if isinstance(game_table, Mapping) else DEFAULT_GAME
    if not isinstance(name, str):
        raise TypeError(f"[game] {GAME_NAME_KEY} must be str, got {type(name).__name__} {name!r}.")
    return name


def experiment_from_tables(tables: Mapping[str, Any]) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from parsed TOML (``{section: {key: value}}``)."""
    for name, table in tables.items():
        if name not in SECTIONS:
            raise ValueError(f"unknown section [{name}]; valid sections: {', '.join(SECTIONS)}.")
        if not isinstance(table, Mapping):
            raise TypeError(f"[{name}] must be a table, got {type(table).__name__}.")
    game_name = game_name_of(tables)
    try:
        types = section_types(game_name)
    except ValueError as error:
        raise ValueError(f"[game] {GAME_NAME_KEY}: {error}") from error
    sections = {}
    for name, table in tables.items():
        if name == "game":
            table = {key: value for key, value in table.items() if key != GAME_NAME_KEY}
        sections[name] = dataclass_from_table(types[name], table, section=name)
    return ExperimentConfig(game_name=game_name, **sections)


def experiment_to_tables(config: ExperimentConfig) -> dict[str, dict[str, Scalar]]:
    """``{section: {key: value}}`` for every section, ``None`` values omitted; ``[game]`` starts with ``name``."""
    tables = {name: dataclass_to_table(getattr(config, name)) for name in SECTIONS}
    tables["game"] = {GAME_NAME_KEY: config.game_name, **tables["game"]}
    return tables


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
