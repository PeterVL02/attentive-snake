"""Building environments and the policy from an ``ExperimentConfig``.

The policy's input shape and action count come from the environment, so build
the environment first and pass it to ``make_policy``.
"""

from typing import Protocol

from torch import nn

from attentive_snake.config import ExperimentConfig
from attentive_snake.games.snake.env import SnakeEnv
from attentive_snake.env.vec_env import SyncVecSnakeEnv
from attentive_snake.models.registry import build_policy


class EnvSpec(Protocol):
    """What the policy needs to know about an environment (``SnakeEnv`` and ``SyncVecSnakeEnv`` both qualify)."""

    observation_shape: tuple[int, int, int]
    num_actions: int
    history_length: int


def make_vec_env(config: ExperimentConfig) -> SyncVecSnakeEnv:
    """The training environments, seeded with ``config.train.seed``."""
    return SyncVecSnakeEnv(
        config.env.num_envs,
        config.game,
        config.reward,
        seed=config.train.seed,
        history_length=config.env.history_length,
        history_actions=config.env.history_actions,
    )


def make_env(config: ExperimentConfig, seed: int | None = None) -> SnakeEnv:
    """A single environment with the experiment's game and reward (e.g. for watching)."""
    return SnakeEnv(
        config.game,
        config.reward,
        seed=seed,
        history_length=config.env.history_length,
        history_actions=config.env.history_actions,
    )


def make_policy(config: ExperimentConfig, env: EnvSpec) -> nn.Module:
    return build_policy(config.model, tuple(env.observation_shape), env.num_actions, env.history_length)
