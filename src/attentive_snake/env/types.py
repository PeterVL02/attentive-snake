"""Contract between games, the generic environment layer and whatever drives them (trainer, watching).

Environments speak numpy only; they know nothing about torch or models.

Layers, inner to outer:

* ``GameEnv`` (one per game, in ``attentive_snake.games.<name>``): one game, one frame per
  observation, no auto-reset.
* ``SingleEnv`` (``env.history_env.HistoryEnv`` wraps a ``GameEnv``): adds frame stacking
  and optional action channels; this is what is watched and what a vector env holds.
* ``VecEnv`` (``env.vec_env.SyncVecEnv`` holds N ``SingleEnv``): auto-reset, per-env seeds,
  episode summaries; this is what the trainer drives.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EpisodeSummary:
    """Statistics of one finished episode, for logging.

    ``score`` is the game's own headline number (food eaten for Snake), ``outcome`` how
    the episode ended in the game's words (Snake: ``"hit_wall"``, ``"hit_self"``,
    ``"won"``, ``"starved"``).
    """

    env_index: int
    score: float
    length: int  # steps taken
    total_reward: float
    outcome: str


@dataclass(frozen=True)
class EnvStep:
    """Result of stepping a single environment.

    ``event`` is what happened in this step in the game's words (Snake: ``"moved"``,
    ``"ate_food"``, ...); on the step that ends an episode it is the episode's outcome.
    ``score`` is the game's headline number for the episode so far.
    """

    observation: np.ndarray  # float32, shape == observation_shape
    reward: float
    terminated: bool  # ended by the game's rules (death, win)
    truncated: bool  # cut short by a limit; treated like termination by the trainer
    event: str
    score: float


class GameEnv(Protocol):
    """What a game provides: one environment, one frame per observation, no auto-reset.

    ``observation_shape`` is one frame: ``(F, H, W)`` for grid and image games, ``(F,)``
    for vector games. ``view`` returns the JSON-able payload of the game's ``view_kind``
    for the current state (see ``attentive_snake.views``).
    """

    observation_shape: tuple[int, ...]
    num_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(self, action: int) -> EnvStep: ...

    def view(self) -> dict[str, Any]: ...


class SingleEnv(GameEnv, Protocol):
    """A ``GameEnv`` whose observations are stacks of the last ``history_length`` frames.

    ``observation_shape`` is the stacked shape (``history_length * frame_channels`` along
    axis 0); each frame ends with ``action_channels`` one-hot action channels (0 = none).
    """

    history_length: int
    frame_channels: int
    action_channels: int


@dataclass(frozen=True)
class VecEnvStep:
    """Result of stepping N environments at once.

    Environments that finished are reset automatically, so ``observations[i]``
    is already the first observation of the next episode when
    ``terminated[i] or truncated[i]``.
    """

    observations: np.ndarray  # float32, (num_envs, *observation_shape)
    rewards: np.ndarray  # float32, (num_envs,)
    terminated: np.ndarray  # bool, (num_envs,)
    truncated: np.ndarray  # bool, (num_envs,)
    finished_episodes: tuple[EpisodeSummary, ...]


class VecEnv(Protocol):
    """What a trainer needs from a batch of parallel environments."""

    num_envs: int
    observation_shape: tuple[int, ...]
    num_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all environments; returns observations (num_envs, *observation_shape)."""
        ...

    def step(self, actions: np.ndarray) -> VecEnvStep:
        """Apply one integer action in ``[0, num_actions)`` per environment."""
        ...
