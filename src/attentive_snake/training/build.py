"""Building the ``Trainer`` of any algorithm from the run's settings."""

from collections.abc import Sequence
from typing import Any

from torch import nn

from attentive_snake.env.types import VecEnv
from attentive_snake.training.algorithm import LearnerSetup
from attentive_snake.training.algorithms import get_algorithm
from attentive_snake.training.config import TrainConfig
from attentive_snake.training.hooks import UpdateHook
from attentive_snake.training.metrics import MetricsLogger
from attentive_snake.training.runtime import resolve_device, seed_everything
from attentive_snake.training.trainer import Trainer


def build_trainer(
    policy: nn.Module,
    env: VecEnv,
    train: TrainConfig,
    settings: Any,
    logger: MetricsLogger | None = None,
    hooks: Sequence[UpdateHook] = (),
) -> Trainer:
    """A trainer of ``train.algorithm`` with its ``settings`` (the algorithm's own section).

    Seeds every RNG from ``train.seed`` first, then moves ``policy`` to the configured
    device in place and builds the learner; the policy's initial weights are whatever the
    caller built (seed before constructing it for fully reproducible runs). The policy
    must have the algorithm's head (``factory.make_policy`` builds it so).
    """
    algorithm = get_algorithm(train.algorithm)
    if not isinstance(settings, algorithm.settings_type):
        raise TypeError(f"{algorithm.name} needs {algorithm.settings_type.__name__} settings, got {type(settings).__name__}.")
    seed_everything(train.seed)
    device = resolve_device(train.device)
    learner = algorithm.build(LearnerSetup(policy.to(device), env, train, settings, device))
    return Trainer(learner, train, device, logger, hooks)
