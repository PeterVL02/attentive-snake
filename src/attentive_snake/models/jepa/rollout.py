"""Unrolling a jepa predictor on its own outputs along the actions that were really taken.

One real step trains the predictor on encoder embeddings only, so a second chained step (what
lookahead needs) gets an input it never saw. With ``Rollout.steps`` = K above 1 the prediction
loss (``models.jepa.objective``) starts from the encoder's embedding of ``s_0`` and feeds the
predictor its own prediction K - 1 times: ``z_1 = predictor(z_0, a_0)``, ``z_2 = predictor(z_1, a_1)``, ...
Every ``z_j`` is compared with the real ``s_j``. K = 1 is the one-step loss.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from attentive_snake.models.auxiliary import TransitionBatch, TransitionSequences


@dataclass(frozen=True)
class Rollout:
    """How far the prediction loss unrolls the predictor, and the weight of the values read from the unrolled embeddings.

    ``value_coefficient`` weights the chained value consistency (``models.jepa.chained_values``); 0 = that term is off.
    """

    steps: int = 1
    value_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if self.value_coefficient < 0.0:
            raise ValueError("value_coefficient must be >= 0.")


def unroll(predictor: nn.Module, embeddings: Tensor, actions: Tensor) -> Tensor:
    """The predictor applied to its own output along ``actions`` (B, K): embeddings (B, ...) -> predictions (B, K, ...).

    Column j is the predicted embedding after actions 0..j. Gradients flow through the whole chain.
    """
    predictions = []
    for step in range(actions.shape[1]):
        embeddings = predictor.step(embeddings, actions[:, step])
        predictions.append(embeddings)
    return torch.stack(predictions, dim=1)


def steps_of(batch: TransitionBatch) -> TransitionSequences:
    """The batch's sequences, or its own transitions as sequences of one step (none of them known to be terminal)."""
    if batch.sequences is not None:
        return batch.sequences
    valid = batch.valid.unsqueeze(1)
    return TransitionSequences(batch.actions.unsqueeze(1), batch.next_observations.unsqueeze(1), valid, torch.zeros_like(valid))
