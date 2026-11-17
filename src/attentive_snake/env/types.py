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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from attentive_snake.env.action_space import ActionSpace


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
    truncated: bool  # cut short by a limit (the trainer may bootstrap from the final observation)
    event: str
    score: float


class GameEnv(Protocol):
    """What a game provides: one environment, one frame per observation, no auto-reset.

    ``observation_shape`` is one frame: ``(F, H, W)`` for grid and image games, ``(F,)``
    for vector games. ``view`` returns the JSON-able payload of the game's ``view_kind``
    for the current state (see ``attentive_snake.views``).

    Optional: an env whose action set depends on its rules (Atari without ``full_action_space``)
    may carry ``action_labels: tuple[str, ...]``, one name per action; watching prefers it over
    ``GameSpec.action_labels`` (``watch.action_names``).

    Optional: an env with continuous actions carries ``action_space: ContinuousActions``
    (``env.action_space``); without it the actions are ``DiscreteActions(num_actions)``. Its
    ``num_actions`` is then the dimension of the action vector, its ``action_labels`` (if any)
    name the dimensions, and ``step`` gets a float32 vector already clipped to the bounds.
    """

    observation_shape: tuple[int, ...]
    num_actions: int  # discrete: the number of actions; continuous: the dimension of the action vector

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(self, action: int | np.ndarray) -> EnvStep: ...

    def view(self) -> dict[str, Any]: ...


class SingleEnv(GameEnv, Protocol):
    """A ``GameEnv`` whose observations are stacks of the last ``history_length`` frames.

    ``observation_shape`` is the stacked shape (``history_length * frame_channels`` along
    axis 0); each frame ends with ``action_channels`` one-hot action channels (0 = none).
    ``action_space`` is always set (``env.action_space.action_space_of`` of the wrapped game);
    ``step`` takes what a policy produced and hands the game ``action_space.env_action`` of it.
    """

    action_space: ActionSpace

    history_length: int
    frame_channels: int
    action_channels: int


@dataclass(frozen=True)
class VecEnvStep:
    """Result of stepping N environments at once.

    Environments that finished are reset automatically, so ``observations[i]``
    is already the first observation of the next episode when
    ``terminated[i] or truncated[i]``. The last observation of the finished episode
    (what the ``SingleEnv``'s ``step`` returned before the reset) is in
    ``final_observations[i]``: a learner bootstraps from it when the episode was
    truncated, since the game would have gone on from there. ``final_observations``
    holds exactly the envs that finished in this step (empty if none did).
    """

    observations: np.ndarray  # float32, (num_envs, *observation_shape)
    rewards: np.ndarray  # float32, (num_envs,)
    terminated: np.ndarray  # bool, (num_envs,)
    truncated: np.ndarray  # bool, (num_envs,)
    finished_episodes: tuple[EpisodeSummary, ...]
    final_observations: Mapping[int, np.ndarray]  # env index -> float32 observation_shape


class VecEnv(Protocol):
    """What a trainer needs from a batch of parallel environments."""

    num_envs: int
    observation_shape: tuple[int, ...]
    num_actions: int
    action_space: ActionSpace

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all environments; returns observations (num_envs, *observation_shape)."""
        ...

    def step(self, actions: np.ndarray) -> VecEnvStep:
        """Apply one action per environment: an integer in ``[0, num_actions)`` (discrete), or a row of
        ``(num_envs, dimension)`` floats (continuous; clipped to the bounds before the game sees it)."""
        ...
