"""Small single-purpose convolutional blocks. Layout: ``(batch, channels, height, width)``."""

from torch import Tensor, nn


class ResidualConv(nn.Module):
    """Same-padded convolution with a skip connection: x + GELU(conv(x))."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.conv(x))


class ConvEncoder(nn.Module):
    """``layers`` same-padded convs: one widening conv + GELU, then residual convs.

    (B, C, H, W) -> (B, channels, H, W); each output cell sees a
    ``layers * (kernel_size - 1) + 1`` wide neighbourhood.
    """

    def __init__(self, in_channels: int, channels: int, layers: int, kernel_size: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size, padding=kernel_size // 2), nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualConv(channels, kernel_size) for _ in range(layers - 1)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.stem(x))


class FlattenReadout(nn.Module):
    """Whole feature map -> one vector: flatten, linear, GELU. (B, C, H, W) -> (B, features).

    Keeps every cell's position (unlike mean pooling, which would dilute the single
    head cell); possible because the grid size is fixed per model.
    """

    def __init__(self, channels: int, height: int, width: int, features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels * height * width, features)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.linear(x.flatten(1)))


class GridToTokens(nn.Module):
    """Every cell's channel vector becomes a token: (B, D, H, W) -> (B, H*W, D), row-major."""

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2)
