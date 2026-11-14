"""Named reward configurations stored as flat TOML files (``<directory>/<name>.toml``).

A file is one reward-dataclass table at top level (no section header). Every function
takes the directory explicitly (one directory per game: ``config.layout.rewards_dir``);
names follow the run-name rules (no path traversal). Loading needs the game's reward
dataclass (``GameSpec.reward_type``). What a scheme pays per situation is the game's
optional ``GameSpec.reward_table``.
"""

import tomllib
from pathlib import Path
from typing import Any

from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.toml_writer import format_value
from attentive_snake.run_dir import validate_run_name

__all__ = ["delete_scheme", "list_schemes", "load_scheme", "save_scheme"]

SUFFIX = ".toml"


def _scheme_path(directory: Path, name: str) -> Path:
    return Path(directory) / f"{validate_run_name(name)}{SUFFIX}"


def list_schemes(directory: Path) -> list[str]:
    """Names of the schemes in ``directory``, alphabetical (empty if the directory is missing)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob(f"*{SUFFIX}") if path.is_file())


def load_scheme[T](directory: Path, name: str, reward_type: type[T]) -> T:
    path = _scheme_path(directory, name)
    with path.open("rb") as file:
        table = tomllib.load(file)
    try:
        return dataclass_from_table(reward_type, table)
    except (ValueError, TypeError) as error:
        raise type(error)(f"{path}: {error}") from error


def save_scheme(directory: Path, name: str, config: Any) -> Path:
    path = _scheme_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {format_value(value)}" for key, value in dataclass_to_table(config).items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def delete_scheme(directory: Path, name: str) -> None:
    _scheme_path(directory, name).unlink()
