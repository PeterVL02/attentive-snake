"""Contract between a policy with an auxiliary training objective (a loss besides the algorithm's own) and the learners.

A policy offers one through a method ``auxiliary_objective() -> AuxiliaryObjective | None``
(``auxiliary_objective(policy)`` below asks for it). Policies without the method, or
returning ``None`` (``jepa`` with ``prediction_coef = 0``), have none, and a learner then
trains exactly as it would without this contract: no extra forward pass, no random numbers.

A learner that supports objectives (``training.auxiliary.AuxiliaryTraining`` does the bookkeeping):

* adds ``objective.loss(batch).loss`` (already weighted by its coefficient) to its own loss
  before ``backward``, for transitions it has (observation, action, next observation);
* calls ``objective.after_step()`` after every optimizer step (e.g. to move a target copy);
* saves ``objective.state_dict()`` with its own checkpoint state and restores it with
  ``load_state_dict``; a checkpoint without it (a fork of a run that had no objective) is
  followed by ``restart()``, which rebuilds the state from the policy's loaded weights;
* reports the means of ``AuxiliaryLoss.metrics`` (keys among the kind's training metrics);
* if the objective wants more than one step (``sequence_length(objective)`` above 1: a model
  unrolled along the steps really taken), also fills ``TransitionBatch.sequences`` with that
  many steps per transition. Objectives without a ``sequence_length`` get today's batch.

The objective's state (e.g. an EMA target encoder) is not part of the policy: the policy's
weights, its optimizer and DQN's target network never contain it, and evaluation and
watching load the policy alone.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn


@dataclass(frozen=True)
class TransitionSequences:
    """The steps really taken from each transition of a batch on: K per row, the same env, oldest first.

    Column 0 is the batch's own transition; column j the step taken j steps later, from the
    next observation of column j - 1.
    """

    actions: Tensor  # int64 (B, K)
    next_observations: Tensor  # float32 (B, K, *obs_shape), what followed each step (an episode's last observation where it ended)
    valid: Tensor  # float32 (B, K), 1.0 where the step is known and belongs to the same episode as column 0; once 0.0 it stays 0.0
    terminal: Tensor  # float32 (B, K), 1.0 where nothing is bootstrapped from the step's next observation (the game did not go on from it)

    def select(self, indices: Tensor) -> "TransitionSequences":
        return TransitionSequences(self.actions[indices], self.next_observations[indices], self.valid[indices], self.terminal[indices])


@dataclass(frozen=True)
class TransitionBatch:
    """Transitions on the training device. ``valid[i]`` is 1.0 where ``next_observations[i]`` really followed the action.

    ``sequences`` is only filled for an objective that asked for more than one step (``sequence_length``).
    """

    observations: Tensor  # float32 (B, *obs_shape)
    actions: Tensor  # int64 (B,)
    next_observations: Tensor  # float32 (B, *obs_shape)
    valid: Tensor  # float32 (B,), 0.0 = the next observation is unknown: left out of the loss
    sequences: TransitionSequences | None = None  # the steps from each transition on (column 0 = the fields above)


@dataclass(frozen=True)
class AuxiliaryLoss:
    loss: Tensor  # scalar, weighted by the objective's coefficient, carries gradients
    metrics: Mapping[str, float]  # detached numbers about this batch, for the training report


class AuxiliaryObjective(Protocol):
    def loss(self, batch: TransitionBatch) -> AuxiliaryLoss: ...

    def after_step(self) -> None:
        """Called after every optimizer step that included ``loss``."""
        ...

    def restart(self) -> None:
        """Rebuild the objective's state from the policy's current weights (no saved state to restore)."""
        ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


def auxiliary_objective(policy: nn.Module) -> AuxiliaryObjective | None:
    """The policy's auxiliary objective, or ``None`` if it has none."""
    make = getattr(policy, "auxiliary_objective", None)
    return make() if callable(make) else None


def sequence_length(objective: AuxiliaryObjective) -> int:
    """How many steps in a row the objective's loss reads per transition (its ``sequence_length`` attribute; 1 without one)."""
    return int(getattr(objective, "sequence_length", 1))
