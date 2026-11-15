"""Value consistency of the jepa kind: how far the value of a predicted next embedding is from the value of the real one.

With ``value_training = "actual"`` the value decoder is trained on real next-state embeddings
but acts on predicted ones, so what decides an action is the decoded prediction. Its error
must be small next to the value differences between actions, which in a game like Lunar
Lander are tiny (one step changes a state little). The embedding-matching loss alone treats
every direction of the embedding alike; the squared value gap below trains the predictor
in the one direction the decoder reads.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from attentive_snake.models.jepa.readout import whole


def frozen_decode(decoder: nn.Module, vectors: Tensor) -> Tensor:
    """The decoder's value of each vector, (..., D) -> (...), with gradients into the vectors but not the decoder's weights."""
    linear = decoder.linear
    return F.linear(vectors, linear.weight.detach(), linear.bias.detach()).squeeze(-1)


def embedding_value_gaps(
    decoder: nn.Module, predictions: Tensor, real_embeddings: Tensor, readout: Callable[[Tensor], Tensor] = whole
) -> Tensor:
    """``decoder(prediction) - decoder(real embedding)`` per row (B,), with gradients into the predictions only."""
    with torch.no_grad():
        real = decoder(readout(real_embeddings))
    return frozen_decode(decoder, readout(predictions)) - real


def value_gaps(
    decoder: nn.Module,
    predictions: Tensor,
    encoder: nn.Module,
    next_observations: Tensor,
    readout: Callable[[Tensor], Tensor] = whole,
) -> Tensor:
    """``decoder(prediction) - decoder(encoder(s'))`` per row (B,), with gradients into the predictions only.

    The decoder reads ``readout`` of both embeddings (``models.jepa.readout``: the whole
    vector, or a token set's class token).

    The decoder's weights are used without gradients (it learns from the TD loss alone, at
    real embeddings) and the real next embedding comes from the online encoder without
    gradients: the encoder the decoder is trained on, not the EMA target.
    """
    with torch.no_grad():
        real_embeddings = encoder(next_observations)
    return embedding_value_gaps(decoder, predictions, real_embeddings, readout)
