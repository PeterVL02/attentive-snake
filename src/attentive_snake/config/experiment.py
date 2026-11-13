"""The complete description of one experiment: game, reward, model, training and env settings.

Pure data. Reading and writing it lives in ``attentive_snake.config.experiment_io``.
"""

from dataclasses import dataclass, field

from attentive_snake.env.reward import RewardConfig
from attentive_snake.game.config import GameConfig
from attentive_snake.models.config import ModelConfig
from attentive_snake.training.config import TrainConfig


@dataclass(frozen=True)
class EnvConfig:
    """How the environments are run (not what the game is). Field descriptions live in ``metadata["help"]``."""

    num_envs: int = field(default=64, metadata={"help": "How many games are played in parallel to collect experience. More games give a bigger, more varied batch per update."})
    history_length: int = field(default=1, metadata={"help": "How many of the most recent frames the model sees at once (1 = only the current board). More frames let it see where the snake has been and which way it is moving."})
    history_actions: bool = field(default=False, metadata={"help": "Also tell the model which action was taken at each past frame. Only useful together with history_length above 1."})

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive.")
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Every setting of a run, one field per TOML section of the same name."""

    game: GameConfig = field(default_factory=GameConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
