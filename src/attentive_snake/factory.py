"""Building environments, the policy and the trainer from an ``ExperimentConfig``, for any game and algorithm.

The game (``config.game_name``) provides one ``GameEnv``; history, action channels and
vectorisation are added here generically. The policy's input layout and action count come
from the environment, so build the environment first and pass it to ``make_policy``.
"""

from collections.abc import Sequence
from typing import Protocol

from torch import nn

from attentive_snake.config import ExperimentConfig
from attentive_snake.env.history_env import HistoryEnv
from attentive_snake.env.types import GameEnv
from attentive_snake.env.vec_env import SyncVecEnv, env_seed
from attentive_snake.games.registry import get_game
from attentive_snake.models.registry import build_policy
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training.algorithms import get_algorithm
from attentive_snake.training.build import build_trainer
from attentive_snake.training.hooks import UpdateHook
from attentive_snake.training.metrics import MetricsLogger
from attentive_snake.training.trainer import Trainer


class EnvSpec(Protocol):
    """What the policy needs to know about an environment (``HistoryEnv`` and ``SyncVecEnv`` both qualify)."""

    observation_shape: tuple[int, ...]
    num_actions: int
    history_length: int
    action_channels: int


def make_game_env(config: ExperimentConfig, seed: int | None = None) -> GameEnv:
    """The game's own environment (one frame per observation) with the experiment's rules and reward."""
    return get_game(config.game_name).make_env(config.game, config.reward, seed)


def make_env(config: ExperimentConfig, seed: int | None = None) -> HistoryEnv:
    """A single environment as the policy sees it (history and action channels per ``[env]``), e.g. for watching."""
    return HistoryEnv(make_game_env(config, seed), config.env.history_length, config.env.history_actions)


def make_vec_env(config: ExperimentConfig) -> SyncVecEnv:
    """The training environments, ``config.env.num_envs`` of them, seeded from ``config.train.seed``."""
    seed = config.train.seed
    return SyncVecEnv([make_env(config, env_seed(seed, i)) for i in range(config.env.num_envs)])


def observation_spec(config: ExperimentConfig, env: EnvSpec) -> ObservationSpec:
    """The input layout of ``env`` for an experiment of ``config``'s game."""
    return ObservationSpec(
        kind=get_game(config.game_name).observation_kind,
        shape=tuple(env.observation_shape),
        history_length=env.history_length,
        action_channels=env.action_channels,
    )


def make_policy(config: ExperimentConfig, env: EnvSpec) -> nn.Module:
    """The policy of ``[model]`` with the head its algorithm trains (``Algorithm.head``)."""
    head = get_algorithm(config.train.algorithm).head
    return build_policy(config.model, observation_spec(config, env), env.num_actions, head)


def make_trainer(
    config: ExperimentConfig,
    policy: nn.Module,
    env: SyncVecEnv,
    logger: MetricsLogger | None = None,
    hooks: Sequence[UpdateHook] = (),
) -> Trainer:
    """The trainer of ``config.train.algorithm`` with the algorithm's section (see ``training.build.build_trainer``)."""
    return build_trainer(policy, env, config.train, config.algorithm_settings, logger, hooks)
