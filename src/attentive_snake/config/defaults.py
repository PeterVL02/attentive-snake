"""The default experiment of a game for a chosen algorithm, from the files of ``config.layout``.

``configs/<game>/default.toml`` is complete for its own algorithm (``[train] algorithm``,
PPO in the shipped files). For another algorithm, the file
``configs/<game>/algorithms/<algorithm>.toml`` is laid over it key by key: it holds that
algorithm's section and the shared ``[train]`` / ``[env]`` / ... values that suit it better
(e.g. the learning rate); default.toml's own algorithm section is dropped. Without that file
the algorithm gets its code defaults. Each value has one source: an algorithm's defaults for
a game live either in default.toml or in its algorithms file, never both.
"""

import tomllib
from pathlib import Path
from typing import Any

from attentive_snake.config.experiment import ExperimentConfig
from attentive_snake.config.experiment_io import ALGORITHM_KEY, algorithm_of, experiment_from_tables
from attentive_snake.config.layout import algorithm_defaults_path, default_config_path


def default_experiment(configs_dir: Path, game: str, algorithm: str | None = None) -> ExperimentConfig:
    """The game's default experiment trained with ``algorithm`` (``None``: default.toml's own algorithm).

    ``OSError`` if default.toml is missing; ``ValueError`` / ``TypeError`` for invalid files.
    """
    path = default_config_path(configs_dir, game)
    tables = _read(path)
    if algorithm is not None and algorithm != algorithm_of(tables):
        tables = _overlay(_without(tables, algorithm_of(tables)), _algorithm_tables(configs_dir, game, algorithm))
    try:
        return experiment_from_tables(tables)
    except (ValueError, TypeError) as error:
        raise type(error)(f"{path}: {error}") from error


def _algorithm_tables(configs_dir: Path, game: str, algorithm: str) -> dict[str, dict[str, Any]]:
    path = algorithm_defaults_path(configs_dir, game, algorithm)
    tables = _read(path) if path.is_file() else {}
    return _overlay(tables, {"train": {ALGORITHM_KEY: algorithm}})


def _read(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def _without(tables: dict[str, dict[str, Any]], section: str) -> dict[str, dict[str, Any]]:
    return {name: table for name, table in tables.items() if name != section}


def _overlay(base: dict[str, dict[str, Any]], top: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = {name: dict(table) for name, table in base.items()}
    for name, table in top.items():
        merged.setdefault(name, {}).update(table)
    return merged
