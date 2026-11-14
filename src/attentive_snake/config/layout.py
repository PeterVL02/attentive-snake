"""Where a game's config files live under a configs directory (``configs/`` in the project)::

    configs/<game>/default.toml               the default experiment of the game (every setting but the rewards)
    configs/<game>/rewards/<name>.toml        named reward schemes; "default" holds the default reward values
    configs/<game>/algorithms/<name>.toml     the game's defaults for another algorithm than default.toml's:
                                              its own section plus the [train] / [env] values that differ
"""

from pathlib import Path

DEFAULT_CONFIG_FILENAME = "default.toml"
REWARDS_DIRNAME = "rewards"
ALGORITHMS_DIRNAME = "algorithms"
DEFAULT_SCHEME = "default"


def game_configs_dir(configs_dir: Path, game: str) -> Path:
    return Path(configs_dir) / game


def default_config_path(configs_dir: Path, game: str) -> Path:
    return game_configs_dir(configs_dir, game) / DEFAULT_CONFIG_FILENAME


def rewards_dir(configs_dir: Path, game: str) -> Path:
    return game_configs_dir(configs_dir, game) / REWARDS_DIRNAME


def algorithm_defaults_path(configs_dir: Path, game: str, algorithm: str) -> Path:
    return game_configs_dir(configs_dir, game) / ALGORITHMS_DIRNAME / f"{algorithm}.toml"
