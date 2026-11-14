"""Value consistency of the jepa kind: how far the value of a predicted next embedding is from the value of the real one.

With ``value_training = "actual"`` the value decoder is trained on real next-state embeddings
but acts on predicted ones, so what decides an action is the decoded prediction. Its error
must be small next to the value differences between actions, which in a game like Lunar
Lander are tiny (one step changes a state little). The embedding-matching loss alone treats
every direction of the embedding alike; the squared value gap below trains the predictor
in the one direction the decoder reads.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def value_gaps(decoder: nn.Module, predictions: Tensor, encoder: nn.Module, next_observations: Tensor) -> Tensor:
    """``decoder(prediction) - decoder(encoder(s'))`` per row (B,), with gradients into the predictions only.

    The decoder's weights are used without gradients (it learns from the TD loss alone, at
    real embeddings) and the real next embedding comes from the online encoder without
    gradients: the encoder the decoder is trained on, not the EMA target.
    """
    with torch.no_grad():
        real = decoder(encoder(next_observations))
    linear = decoder.linear
    predicted = F.linear(predictions, linear.weight.detach(), linear.bias.detach()).squeeze(-1)
    return predicted - real
