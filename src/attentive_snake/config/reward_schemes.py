"""Named reward configurations stored as flat TOML files (``<directory>/<name>.toml``).

A file is one ``RewardConfig`` table at top level (no section header). Every function
takes the directory explicitly; names follow the run-name rules (no path traversal).
``reward_table`` (what a scheme pays per situation) lives in ``reward_table.py`` and is re-exported here.
"""

import tomllib
from pathlib import Path

from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.toml_writer import format_value
from attentive_snake.games.snake.reward_table import reward_table
from attentive_snake.games.snake.reward import RewardConfig
from attentive_snake.run_dir import validate_run_name

__all__ = ["delete_scheme", "list_schemes", "load_scheme", "reward_table", "save_scheme"]

SUFFIX = ".toml"


def _scheme_path(directory: Path, name: str) -> Path:
    return Path(directory) / f"{validate_run_name(name)}{SUFFIX}"


def list_schemes(directory: Path) -> list[str]:
    """Names of the schemes in ``directory``, alphabetical (empty if the directory is missing)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob(f"*{SUFFIX}") if path.is_file())


def load_scheme(directory: Path, name: str) -> RewardConfig:
    path = _scheme_path(directory, name)
    with path.open("rb") as file:
        table = tomllib.load(file)
    try:
        return dataclass_from_table(RewardConfig, table)
    except (ValueError, TypeError) as error:
        raise type(error)(f"{path}: {error}") from error


def save_scheme(directory: Path, name: str, config: RewardConfig) -> Path:
    path = _scheme_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {format_value(value)}" for key, value in dataclass_to_table(config).items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def delete_scheme(directory: Path, name: str) -> None:
    _scheme_path(directory, name).unlink()
