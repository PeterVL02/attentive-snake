"""Residual blocks: ``x + F(x)`` with ``F`` two layers deep, and the encoders that stack them.

Grids use two same-padded convolutions per block (He et al., 2015, without
normalisation by default: the batch statistics of an RL rollout are a poor fit; ``norm``
adds a batch-independent one after each convolution, ``blocks.norm.map_norm``). Vectors use
two linear layers with a LayerNorm in front of the block's body, as in SimBa
(arXiv 2410.09754). Neither block changes the width, so the skips need no projection.
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.norm import map_norm, normalised, present
from attentive_snake.models.blocks.vector import ElementEmbedding, ElementPositionalEmbedding


class ResidualConvBlock(nn.Module):
    """``GELU(x + conv2(GELU(conv1(x))))``: two same-padded convolutions and a skip. (B, C, H, W) -> same.

    With ``norm`` each convolution's output is normalised: ``GELU(x + norm2(conv2(GELU(norm1(conv1(x))))))``.
    """

    def __init__(self, channels: int, kernel_size: int, norm: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = map_norm(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = map_norm(channels, norm)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.act(normalised(self.norm1, self.conv1(x)))
        return self.act(x + normalised(self.norm2, self.conv2(hidden)))


class ResidualMlpBlock(nn.Module):
    """``x + linear2(GELU(linear1(LayerNorm(x))))`` on the last axis. (B, W) -> same."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear2(self.act(self.linear1(self.norm(x))))


class ResidualConvEncoder(nn.Module):
    """One widening conv + GELU, then ``blocks`` residual conv blocks. (B, C, H, W) -> (B, channels, H, W).

    With ``norm`` every convolution is followed by a normalisation of its feature map.
    """

    def __init__(self, in_channels: int, channels: int, blocks: int, kernel_size: int, norm: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size, padding=kernel_size // 2), *present(map_norm(channels, norm)), nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels, kernel_size, norm) for _ in range(blocks)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x))


class ResidualMlpEncoder(nn.Module):
    """Flatten, one linear embedding, ``blocks`` residual MLP blocks, a final LayerNorm. (B, *shape) -> (B, width)."""

    def __init__(self, in_features: int, width: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Linear(in_features, width)
        self.blocks = nn.Sequential(*[ResidualMlpBlock(width) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.blocks(self.stem(x.flatten(1))))


class ResidualElementStem(nn.Module):
    """Per-element residual MLP: (N, V*E) -> (N, E, width) tokens, one per element.

    The vector counterpart of ``ElementMlpStem`` with ``ResidualMlpBlock``s (shared by all elements)
    after the value + element embedding, then a LayerNorm; token ``e`` still describes element ``e``.
    """

    def __init__(self, values: int, elements: int, width: int, blocks: int) -> None:
        super().__init__()
        self.embed = ElementEmbedding(values, width)
        self.positional = ElementPositionalEmbedding(elements, width)
        self.blocks = nn.Sequential(*[ResidualMlpBlock(width) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)

    def forward(self, numbers: Tensor) -> Tensor:
        return self.norm(self.blocks(self.positional(self.embed(numbers))))
