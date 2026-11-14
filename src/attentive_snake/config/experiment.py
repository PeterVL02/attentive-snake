"""The complete description of one experiment: which game, its rules and rewards, and the model, training, algorithm, env and evaluation settings.

Pure data. Reading and writing it lives in ``attentive_snake.config.experiment_io``.
``[game]`` and ``[reward]`` are the dataclasses of the chosen game (``GameSpec.rules_type``
and ``reward_type``); the algorithm's section is named like the algorithm (``[train]
algorithm``: ``[ppo]``, ``[dqn]``) and holds its ``Algorithm.settings_type``; ``[model]
[train] [env] [evaluation]`` are the same for every game and algorithm.
"""

from dataclasses import dataclass, field
from typing import Any

from attentive_snake.evaluation.config import EvalConfig
from attentive_snake.games.registry import DEFAULT_GAME, get_game
from attentive_snake.models.config import ModelConfig
from attentive_snake.training.algorithms import get_algorithm
from attentive_snake.training.config import DEFAULT_ALGORITHM, TrainConfig

SHARED_SECTIONS: tuple[str, ...] = ("game", "reward", "model", "train", "env", "evaluation")
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
    be instances of the game's ``rules_type`` / ``reward_type``. ``algorithm_settings`` is
    the ``[<train.algorithm>]`` section: ``None`` becomes the algorithm's defaults, a given
    value must be an instance of its ``settings_type``.
    """

    game_name: str = DEFAULT_GAME
    game: Any = None
    reward: Any = None
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    algorithm_settings: Any = None

    def __post_init__(self) -> None:
        types = section_types(self.game_name, self.train.algorithm)
        for attribute, section in (("game", "game"), ("reward", "reward"), ("algorithm_settings", self.train.algorithm)):
            value = getattr(self, attribute)
            if value is None:
                object.__setattr__(self, attribute, types[section]())
            elif not isinstance(value, types[section]):
                raise TypeError(
                    f"[{section}] of a {self.game_name} {self.train.algorithm} experiment must be a "
                    f"{types[section].__name__}, got {type(value).__name__}."
                )

    def sections(self) -> dict[str, Any]:
        """``{section name: its dataclass}`` in file order (``[game]`` without its ``name``)."""
        return {
            name: self.algorithm_settings if name == self.train.algorithm else getattr(self, name)
            for name in section_names(self.train.algorithm)
        }


def section_names(algorithm: str = DEFAULT_ALGORITHM) -> tuple[str, ...]:
    """The sections of an experiment that trains with ``algorithm``, in file order (its own section after ``[train]``)."""
    shared = list(SHARED_SECTIONS)
    shared.insert(shared.index("train") + 1, algorithm)
    return tuple(shared)


def section_types(game_name: str, algorithm: str = DEFAULT_ALGORITHM) -> dict[str, type]:
    """The dataclass of every section for an experiment of ``game_name`` trained with ``algorithm``.

    ``ValueError`` for an unknown game or algorithm.
    """
    game = get_game(game_name)
    return {
        "game": game.rules_type,
        "reward": game.reward_type,
        "model": ModelConfig,
        "train": TrainConfig,
        algorithm: get_algorithm(algorithm).settings_type,
        "env": EnvConfig,
        "evaluation": EvalConfig,
    }
