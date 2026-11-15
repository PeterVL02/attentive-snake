"""The JEPA prediction loss: the predicted embedding of the next state must match the target encoder's embedding of it.

For each transition ``(s, a, s')``: ``z = encoder(s)`` (online, with gradients),
``prediction = predictor(z, a)``, ``target = target_encoder(s')`` (an EMA copy of the
encoder, no gradients). The distance depends on what the value decoder reads
(``ModelConfig.value_training``):

* ``"predicted"``: the cosine distance ``1 - cos(prediction, target)``, i.e. half the squared
  distance of the two embeddings scaled to unit length. Scale-free on purpose: the encoder
  cannot lower it by shrinking its embeddings, only by making the next state predictable;
  the TD loss on the predicted embedding sets the scale the value decoder needs.
* ``"actual"``: the mean squared distance of the embeddings themselves. The value decoder is
  trained on real embeddings there and applied to predicted ones when acting, so a
  prediction must match the real embedding in length as well as direction. A second term,
  value consistency (``models.jepa.value_consistency``), also makes the decoded prediction
  match the decoded real next embedding: the squared distance treats every direction of the
  embedding alike, while acting needs the prediction right in the one direction the decoder reads.

``trains_encoder = False`` stops the loss at the embedding ``z``: it then trains the predictor
only and the encoder learns from the algorithm's loss alone.

The slowly moving target (as in BYOL / I-JEPA) is what keeps the encoder
from the trivial solution of mapping every state to the same embedding; the embedding
spread (``embedding_std``) is logged to show whether that happened anyway.

Shape-agnostic: an embedding may be one vector per state (B, D) or a token set (B, T, D)
(``jepa_tokens``). Distances and spread are then taken per token and averaged over the tokens;
the value gap reads ``readout`` of the embeddings (``models.jepa.readout``).
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from attentive_snake.models.auxiliary import AuxiliaryLoss, TransitionBatch
from attentive_snake.models.jepa.ema import EmaCopy
from attentive_snake.models.jepa.readout import whole
from attentive_snake.models.jepa.value_consistency import value_gaps


def per_row(values: Tensor) -> Tensor:
    """(B, ...) -> (B,): the mean over every axis but the first (a (B,) tensor is returned as it is)."""
    return values if values.ndim == 1 else values.flatten(1).mean(dim=1)


def cosine_distance(predictions: Tensor, targets: Tensor) -> Tensor:
    """``1 - cos`` per row: (B, D), (B, D) -> (B,), in [0, 2]; for token sets (B, T, D) per token, averaged over the tokens."""
    return per_row(1.0 - F.cosine_similarity(predictions, targets, dim=-1))


def squared_distance(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared difference per row: (B, D), (B, D) -> (B,); for token sets (B, T, D) over every token's numbers."""
    return per_row((predictions - targets).pow(2).mean(dim=-1))


DISTANCES = {"predicted": cosine_distance, "actual": squared_distance}  # by ModelConfig.value_training


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` where ``mask`` is 1 (0 if nothing is)."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def embedding_spread(embeddings: Tensor) -> Tensor:
    """Mean over dimensions (and tokens, for token sets) of the batch standard deviation of the unit-length embeddings.

    0 when every state gets the same embedding (collapse); about ``1 / sqrt(d_model)`` when
    the embeddings point in unrelated directions.
    """
    return F.normalize(embeddings, dim=-1).std(dim=0, correction=0).mean()


class PredictionObjective:
    """The ``AuxiliaryObjective`` of the ``jepa`` kind; owns the EMA target encoder.

    ``encoder`` is the online encoder (it shares its modules with the policy), ``predictor``
    the policy's predictor; ``coefficient`` weights the loss, ``decay`` is the EMA rate,
    ``distance`` compares predictions with targets row by row (``DISTANCES``),
    ``trains_encoder`` whether the loss reaches the encoder. With a ``value_decoder`` the mean
    value gap (``value_consistency.value_gaps``, the decoder reading ``readout`` of the
    embeddings) is reported, and with ``consistent_values`` (``"actual"``) its square is added
    to the loss. ``predictor`` is anything with ``step(embeddings, actions)`` that keeps the
    embedding's shape (``blocks.predictor``, ``blocks.token_predictor``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        coefficient: float,
        decay: float,
        distance: Callable[[Tensor, Tensor], Tensor] = cosine_distance,
        trains_encoder: bool = True,
        value_decoder: nn.Module | None = None,
        consistent_values: bool = False,
        readout: Callable[[Tensor], Tensor] = whole,
    ) -> None:
        self._encoder = encoder
        self._readout = readout
        self._distance = distance
        self._trains_encoder = trains_encoder
        self._value_decoder = value_decoder
        self._consistent_values = consistent_values
        self._predictor = predictor
        self._coefficient = coefficient
        self.target = EmaCopy(encoder, decay)

    def loss(self, batch: TransitionBatch) -> AuxiliaryLoss:
        embeddings = self._encoder(batch.observations)
        if not self._trains_encoder:
            embeddings = embeddings.detach()
        predictions = self._predictor.step(embeddings, batch.actions)
        targets = self.target(batch.next_observations)
        loss = masked_mean(self._distance(predictions, targets), batch.valid)
        metrics = {"prediction_loss": float(loss.detach()), "embedding_std": float(embedding_spread(embeddings))}
        if self._value_decoder is not None:
            gaps = value_gaps(self._value_decoder, predictions, self._encoder, batch.next_observations, self._readout)
            metrics["value_gap"] = float(masked_mean(gaps.detach().abs(), batch.valid))
            if self._consistent_values:
                loss = loss + masked_mean(gaps.pow(2), batch.valid)
        return AuxiliaryLoss(loss=self._coefficient * loss, metrics=metrics)

    def after_step(self) -> None:
        self.target.update()

    def restart(self) -> None:
        self.target.reset()

    def state_dict(self) -> dict[str, Any]:
        return {"target_encoder": self.target.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.target.load_state_dict(state["target_encoder"])
