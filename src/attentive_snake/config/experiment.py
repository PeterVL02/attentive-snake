"""The complete description of one experiment: which game, its rules and rewards, and the model, training and env settings.

Pure data. Reading and writing it lives in ``attentive_snake.config.experiment_io``.
``[game]`` and ``[reward]`` are the dataclasses of the chosen game (``GameSpec.rules_type``
and ``reward_type``); ``[model] [train] [env]`` are the same for every game.
"""

from dataclasses import dataclass, field
from typing import Any

from attentive_snake.games.registry import DEFAULT_GAME, get_game
from attentive_snake.models.config import ModelConfig
from attentive_snake.training.config import TrainConfig

SECTIONS: tuple[str, ...] = ("game", "reward", "model", "train", "env")
GAME_NAME_KEY = "name"  # the key of [game] that selects the game; the other keys are its rules


@dataclass(frozen=True)
class EnvConfig:
    """How the environments are run (not what the game is). Field descriptions live in ``metadata["help"]``."""

    num_envs: int = field(default=64, metadata={"help": "How many games are played in parallel to collect experience. More games give a bigger, more varied batch per update."})
    history_length: int = field(default=1, metadata={"help": "How many of the most recent frames the model sees at once (1 = only the current one). More frames let it see how things have been moving."})
    history_actions: bool = field(default=False, metadata={"help": "Also tell the model which action was taken at each past frame. Only useful together with history_length above 1."})

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive.")
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Every setting of a run.

    ``game_name`` is ``[game] name`` in TOML; ``game`` holds the rest of ``[game]``.
    ``game`` / ``reward`` left as ``None`` become the game's defaults; given values must
    be instances of the game's ``rules_type`` / ``reward_type``.
    """

    game_name: str = DEFAULT_GAME
    game: Any = None
    reward: Any = None
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self) -> None:
        types = section_types(self.game_name)
        for section in ("game", "reward"):
            value = getattr(self, section)
            if value is None:
                object.__setattr__(self, section, types[section]())
            elif not isinstance(value, types[section]):
                raise TypeError(
                    f"[{section}] of a {self.game_name} experiment must be a {types[section].__name__}, "
                    f"got {type(value).__name__}."
                )


def section_types(game_name: str) -> dict[str, type]:
    """The dataclass of every section for an experiment of ``game_name`` (``ValueError`` for an unknown game)."""
    game = get_game(game_name)
    return {"game": game.rules_type, "reward": game.reward_type, "model": ModelConfig, "train": TrainConfig, "env": EnvConfig}
