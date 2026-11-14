"""A learner's side of a policy's auxiliary objective (``models.auxiliary``): the loss term, the after-step call, its checkpoint state and report.

Learners hold ``AuxiliaryTraining.of(policy)``, which is ``None`` for policies without an
objective: they then skip every line about it, so their training is unchanged.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from torch import Tensor, nn

from attentive_snake.models.auxiliary import AuxiliaryObjective, TransitionBatch, auxiliary_objective

STATE_KEY = "auxiliary"  # where the objective's state sits in a learner's extra_state


class AuxiliaryTraining:
    """Wraps one ``AuxiliaryObjective``: weighted loss terms, means of its metrics since the last report, its state."""

    def __init__(self, objective: AuxiliaryObjective) -> None:
        self._objective = objective
        self._window: list[Mapping[str, float]] = []

    @classmethod
    def of(cls, policy: nn.Module) -> "AuxiliaryTraining | None":
        objective = auxiliary_objective(policy)
        return None if objective is None else cls(objective)

    def loss(self, batch: TransitionBatch) -> Tensor:
        """The weighted loss term to add before ``backward``; its metrics are kept for ``report``."""
        terms = self._objective.loss(batch)
        self._window.append(terms.metrics)
        return terms.loss

    def after_step(self) -> None:
        self._objective.after_step()

    def report(self) -> dict[str, float]:
        """Means of the metrics since the previous report (empty if no loss was computed)."""
        window, self._window = self._window, []
        if not window:
            return {}
        return {key: float(np.mean([metrics[key] for metrics in window])) for key in window[0]}

    def save_into(self, extra_state: dict[str, Any]) -> dict[str, Any]:
        """``extra_state`` with the objective's state added under ``STATE_KEY``."""
        return extra_state | {STATE_KEY: self._objective.state_dict()}

    def load_from(self, extra_state: Mapping[str, Any]) -> None:
        """Restore the objective's state, or rebuild it from the policy's weights if the checkpoint has none (a fork of a run without it)."""
        if STATE_KEY in extra_state:
            self._objective.load_state_dict(extra_state[STATE_KEY])
        else:
            self._objective.restart()
