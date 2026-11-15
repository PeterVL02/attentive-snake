"""Small single-purpose convolutional blocks. Layout: ``(batch, channels, height, width)``.

Every block takes ``norm`` (default off): a batch-independent normalisation after its
convolution / linear layer and before the GELU (``blocks.norm``).
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.norm import map_norm, normalised, present, vector_norm


class ResidualConv(nn.Module):
    """Same-padded convolution with a skip connection: x + GELU(conv(x)); with ``norm``, x + GELU(norm(conv(x)))."""

    def __init__(self, channels: int, kernel_size: int, norm: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = map_norm(channels, norm)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(normalised(self.norm, self.conv(x)))


class ConvEncoder(nn.Module):
    """``layers`` same-padded convs: one widening conv + GELU, then residual convs.

    (B, C, H, W) -> (B, channels, H, W); each output cell sees a
    ``layers * (kernel_size - 1) + 1`` wide neighbourhood.
    """

    def __init__(self, in_channels: int, channels: int, layers: int, kernel_size: int, norm: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size, padding=kernel_size // 2), *present(map_norm(channels, norm)), nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualConv(channels, kernel_size, norm) for _ in range(layers - 1)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x))


class FlattenReadout(nn.Module):
    """Whole feature map -> one vector: flatten, linear, GELU. (B, C, H, W) -> (B, features).

    Keeps every cell's position (unlike mean pooling, which would dilute the single
    head cell); possible because the grid size is fixed per model. With ``norm`` the
    vector is layer-normalised before the GELU.
    """

    def __init__(self, channels: int, height: int, width: int, features: int, norm: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(channels * height * width, features)
        self.norm = vector_norm(features, norm)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(normalised(self.norm, self.linear(x.flatten(1))))


class GridToTokens(nn.Module):
    """Every cell's channel vector becomes a token: (B, D, H, W) -> (B, H*W, D), row-major."""

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2)
