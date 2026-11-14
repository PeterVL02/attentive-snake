"""Reading and writing an ``ExperimentConfig`` as a TOML file.

One TOML table per section: ``[game] [reward] [model] [train] [<algorithm>] [env] [evaluation]``.
``[game] name`` selects the game (missing: ``"snake"``); the other ``[game]`` keys and all
of ``[reward]`` are read with that game's dataclasses. ``[train] algorithm`` selects the
algorithm (missing: ``"ppo"``), whose own section is named like it (``[ppo]``, ``[dqn]``);
the section of another algorithm is an error. Missing sections and keys take the dataclass
defaults (so configs without ``[evaluation]`` load with evaluation off); unknown sections
or keys are errors.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.experiment import GAME_NAME_KEY, ExperimentConfig, section_names, section_types
from attentive_snake.config.toml_writer import Scalar, format_tables
from attentive_snake.games.registry import DEFAULT_GAME
from attentive_snake.training.algorithms import algorithm_names
from attentive_snake.training.config import DEFAULT_ALGORITHM

ALGORITHM_KEY = "algorithm"  # the key of [train] that selects the algorithm and so its section


def game_name_of(tables: Mapping[str, Any]) -> str:
    """The game an experiment's tables are for (``[game] name``, default ``"snake"``)."""
    game_table = tables.get("game", {})
    name = game_table.get(GAME_NAME_KEY, DEFAULT_GAME) if isinstance(game_table, Mapping) else DEFAULT_GAME
    if not isinstance(name, str):
        raise TypeError(f"[game] {GAME_NAME_KEY} must be str, got {type(name).__name__} {name!r}.")
    return name


def algorithm_of(tables: Mapping[str, Any]) -> str:
    """The algorithm an experiment's tables train with (``[train] algorithm``, default ``"ppo"``)."""
    train_table = tables.get("train", {})
    name = train_table.get(ALGORITHM_KEY, DEFAULT_ALGORITHM) if isinstance(train_table, Mapping) else DEFAULT_ALGORITHM
    if not isinstance(name, str):
        raise TypeError(f"[train] {ALGORITHM_KEY} must be str, got {type(name).__name__} {name!r}.")
    return name


def experiment_from_tables(tables: Mapping[str, Any]) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from parsed TOML (``{section: {key: value}}``)."""
    for name, table in tables.items():
        if not isinstance(table, Mapping):
            raise TypeError(f"[{name}] must be a table, got {type(table).__name__}.")
    game_name, algorithm = game_name_of(tables), algorithm_of(tables)
    valid = section_names(algorithm)
    for name in tables:
        if name in algorithm_names() and name != algorithm:
            raise ValueError(f"[{name}] holds settings of the {name} algorithm, but [train] {ALGORITHM_KEY} is {algorithm!r}.")
        if name not in valid:
            raise ValueError(f"unknown section [{name}]; valid sections: {', '.join(valid)}.")
    try:
        types = section_types(game_name, algorithm)
    except ValueError as error:
        key = f"[game] {GAME_NAME_KEY}" if algorithm in algorithm_names() else f"[train] {ALGORITHM_KEY}"
        raise ValueError(f"{key}: {error}") from error
    sections = {}
    for name, table in tables.items():
        if name == "game":
            table = {key: value for key, value in table.items() if key != GAME_NAME_KEY}
        sections["algorithm_settings" if name == algorithm else name] = dataclass_from_table(types[name], table, section=name)
    return ExperimentConfig(game_name=game_name, **sections)


def experiment_to_tables(config: ExperimentConfig) -> dict[str, dict[str, Scalar]]:
    """``{section: {key: value}}`` for every section, ``None`` values omitted; ``[game]`` starts with ``name``."""
    tables = {name: dataclass_to_table(value) for name, value in config.sections().items()}
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
