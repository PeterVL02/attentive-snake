"""The contract a training algorithm satisfies (PPO, DQN, ...), so the training loop, the config and the app need not know which one runs.

An ``Algorithm`` names its settings dataclass (its own TOML section, named like the
algorithm), the policy head it trains (``attentive_snake.models.heads``; a function of its
settings, so a setting may choose between head forms) for discrete actions and, if it can train
them at all, for continuous ones (``continuous_head``; ``head_for`` picks by the game's action
space), which of its settings a continued run must keep, the metrics it reports and a builder
of its ``Learner``. The shared loop (``training.trainer.Trainer``)
drives a ``Learner`` iteration by iteration and handles the learning-rate schedule,
logging, checkpoints and hooks for every algorithm alike.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn

from attentive_snake.env.types import EpisodeSummary, VecEnv
from attentive_snake.training.config import TrainConfig
from attentive_snake.training_metric import TrainingMetric

__all__ = ["Algorithm", "Learner", "LearnerSetup", "TrainingMetric"]


@dataclass(frozen=True)
class LearnerSetup:
    """What a learner is built from. ``policy`` already sits on ``device``; seeding is done."""

    policy: nn.Module
    env: VecEnv
    train: TrainConfig
    settings: Any  # the algorithm's ``settings_type``
    device: torch.device


class Learner(Protocol):
    """One algorithm's training state; ``optimizer`` is the one the shared learning-rate schedule sets."""

    policy: nn.Module
    optimizer: torch.optim.Optimizer

    @property
    def steps_per_iteration(self) -> int:
        """Env steps (summed over the envs) one ``iterate`` consumes; constant for a run."""
        ...

    def iterate(self, env_steps: int) -> tuple[EpisodeSummary, ...]:
        """Collect experience and learn from it once; ``env_steps`` = steps taken before. Returns the finished episodes."""
        ...

    def report(self) -> dict[str, float]:
        """The algorithm's metrics since the previous report (keys among its ``Algorithm.metrics``)."""
        ...

    def extra_state(self) -> dict[str, Any]:
        """What a checkpoint holds besides the policy, the optimizer and the step count (e.g. a target network)."""
        ...

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        """Restore what ``extra_state`` returned."""
        ...


@dataclass(frozen=True)
class Algorithm:
    """A selectable training algorithm (``[train] algorithm = name``; its settings are the ``[name]`` section)."""

    name: str
    label: str
    description: str
    settings_type: type
    head: Callable[[Any], str]  # its settings -> the policy head it trains, a key of attentive_snake.models.heads.HEADS
    metrics: tuple[TrainingMetric, ...]
    build: Callable[[LearnerSetup], Learner]
    locked_settings: Mapping[str, str] = field(default_factory=dict)  # field of its section -> why a fork must keep it
    continuous_head: Callable[[Any], str] | None = None  # the head for continuous actions; None: it trains discrete actions only

    @property
    def trains_continuous_actions(self) -> bool:
        return self.continuous_head is not None

    def head_for(self, settings: Any, continuous: bool) -> str:
        """The head this algorithm trains with ``settings`` on a game whose actions are ``continuous`` or not.

        ``ValueError`` for continuous actions when it has no head for them (config validation refuses that earlier).
        """
        if not continuous:
            return self.head(settings)
        if self.continuous_head is None:
            raise ValueError(f"{self.label} trains discrete actions only; this game's actions are continuous.")
        return self.continuous_head(settings)
