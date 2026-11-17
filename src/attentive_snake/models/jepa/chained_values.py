"""Chained value consistency: the Q-values read from a predicted embedding should be those of the real state.

Lookahead with the predictor values a state it never observed: for a predicted embedding
``z`` it reads ``Q(z, a') = value_decoder(readout(predictor(z)))[a']``, the predictor applied to
its own output. Trained on real embeddings only, that reading was off by more than the
difference between actions (README, "Lookahead through the predictor").
``chained_value_gaps`` is that reading minus the Q-values the policy gives the real state the
embedding stands for.

What the gap trains, and what it must not:

* The real state's Q-values are a fixed target (no gradients): they come from the online
  encoder, predictor and decoder, i.e. exactly what the policy acts on and the TD loss trains.
* The predicted embedding is read without gradients into it. The term therefore trains one
  thing: the predictor's step *from* a predicted embedding. It does not move the embedding
  itself (the embedding loss decides where a prediction belongs), and so never reaches the
  earlier steps of the chain or the encoder, which could otherwise lower the gap by making
  states look alike.
* The decoder's weights are used without gradients, as in ``value_consistency``: the decoder
  learns from the TD loss alone, at the embeddings the policy acts on; fitting it to
  out-of-distribution inputs would shift the very values taken as the target.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from attentive_snake.models.jepa.readout import whole
from attentive_snake.models.jepa.value_consistency import frozen_decode


def chained_value_gaps(
    decoder: nn.Module,
    predictor: nn.Module,
    predictions: Tensor,
    real_embeddings: Tensor,
    readout: Callable[[Tensor], Tensor] = whole,
) -> Tensor:
    """``Q(prediction, a') - Q(real state, a')`` for every action: (N, ...), (N, ...) -> (N, A), gradients into the predictor only.

    ``real_embeddings`` are the online encoder's embeddings of the states the predictions stand for.
    """
    with torch.no_grad():
        real = decoder(readout(predictor(real_embeddings)))
    return frozen_decode(decoder, readout(predictor(predictions.detach()))) - real
