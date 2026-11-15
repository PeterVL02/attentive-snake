"""Small single-purpose blocks for vector observations (a flat list of numbers per frame).

Layouts: numbers ``(batch, values * elements)``, element tokens ``(batch, elements, width)``.
"values" are stacked copies of one ``elements``-long vector (history frames), oldest first,
so element ``e`` of copy ``v`` sits at index ``v * elements + e``.

The layers take ``norm`` (default off): a layer norm after the linear layer and before the
GELU (``blocks.norm``).
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks.norm import normalised, present, vector_norm


class ElementEmbedding(nn.Module):
    """Each element's values become one token: (N, V*E) -> (N, E, D). A value embedding of the element's V numbers."""

    def __init__(self, values: int, d_model: int) -> None:
        super().__init__()
        self.values = values
        self.proj = nn.Linear(values, d_model)

    def forward(self, numbers: Tensor) -> Tensor:
        return self.proj(numbers.unflatten(1, (self.values, -1)).transpose(1, 2))


class ElementPositionalEmbedding(nn.Module):
    """Adds a learned embedding per element (which number of the observation a token is): (N, E, D) -> (N, E, D)."""

    def __init__(self, elements: int, d_model: int) -> None:
        super().__init__()
        self.element = nn.Parameter(torch.randn(elements, d_model) * 0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.element


class ResidualLinear(nn.Module):
    """Linear layer with a skip connection: x + GELU(linear(x)), on the last axis; with ``norm``, x + GELU(norm(linear(x)))."""

    def __init__(self, width: int, norm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = vector_norm(width, norm)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(normalised(self.norm, self.linear(x)))


class MlpEncoder(nn.Module):
    """Flatten, then ``layers`` fully connected layers: one widening layer + GELU, then residual layers.

    (B, *shape) -> (B, width). The vector counterpart of ``ConvEncoder``: every output
    feature can depend on every input number, since a list of numbers has no neighbourhoods.
    """

    def __init__(self, in_features: int, width: int, layers: int, norm: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_features, width), *present(vector_norm(width, norm)), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualLinear(width, norm) for _ in range(layers - 1)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x.flatten(1)))


class LinearReadout(nn.Module):
    """Features -> ``features`` wide summary: linear, GELU. (B, C) -> (B, features)."""

    def __init__(self, channels: int, features: int, norm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, features)
        self.norm = vector_norm(features, norm)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(normalised(self.norm, self.linear(x)))


class ElementMlpStem(nn.Module):
    """Per-element MLP: (N, V*E) -> (N, E, width) tokens, one per element.

    Each token starts as the element's value embedding plus a learned element embedding
    and passes through ``layers - 1`` residual layers shared by all elements, so token
    ``e`` still describes element ``e`` (attention over them stays attention over elements).
    With ``norm`` the embedded token is layer-normalised before its GELU, as is every residual layer.
    """

    def __init__(self, values: int, elements: int, width: int, layers: int, norm: bool = False) -> None:
        super().__init__()
        self.embed = ElementEmbedding(values, width)
        self.positional = ElementPositionalEmbedding(elements, width)
        self.norm = vector_norm(width, norm)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(*[ResidualLinear(width, norm) for _ in range(layers - 1)])

    def forward(self, numbers: Tensor) -> Tensor:
        return self.blocks(self.act(normalised(self.norm, self.positional(self.embed(numbers)))))
