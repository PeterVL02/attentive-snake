"""Blocks that read screen pictures ``(B, C, H, W)``: far more pixels than a board has cells, so they shrink the map.

``DownsamplingConvEncoder`` is the convolutional stack of the DQN paper (Mnih et al. 2015):
8x8 stride 4, 4x4 stride 2, 3x3 stride 1, ReLU after each, no padding. ``PatchEmbedding`` cuts
the picture into square patches and makes one token of each (as a vision transformer does).
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.norm import map_norm, present

DOWNSAMPLING_LAYERS: tuple[tuple[int, int], ...] = ((8, 4), (4, 2), (3, 1))  # (kernel, stride) per convolution


def downsampled_size(size: int) -> int:
    """Side length of ``DownsamplingConvEncoder``'s feature map for a picture side of ``size`` pixels (84 -> 7)."""
    for kernel, stride in DOWNSAMPLING_LAYERS:
        size = (size - kernel) // stride + 1
    if size < 1:
        raise ValueError("the picture is too small for the image convolutions (36 pixels a side at least).")
    return size


def patch_count(size: int, patch: int) -> int:
    """Patches of ``patch`` pixels along a picture side of ``size`` pixels (a remainder at the edge is dropped)."""
    if size < patch:
        raise ValueError(f"the picture is smaller than one {patch}-pixel patch.")
    return size // patch


class DownsamplingConvEncoder(nn.Module):
    """(B, C, H, W) -> (B, channels, h, w) with ``h = downsampled_size(H)``: three strided convolutions,
    ``channels // 2`` wide first, then ``channels`` (32, 64, 64 for ``channels = 64``). With ``norm`` each
    convolution is followed by a normalisation of the whole feature map (``blocks.norm``)."""

    def __init__(self, in_channels: int, channels: int, norm: bool = False) -> None:
        super().__init__()
        widths = (max(channels // 2, 1), channels, channels)
        layers: list[nn.Module] = []
        for (kernel, stride), width in zip(DOWNSAMPLING_LAYERS, widths):
            layers += [nn.Conv2d(in_channels, width, kernel, stride), *present(map_norm(width, norm)), nn.ReLU()]
            in_channels = width
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PatchEmbedding(nn.Module):
    """Each ``patch`` x ``patch`` square of the picture becomes one token: (B, C, H, W) -> (B, rows * cols, D), row-major."""

    def __init__(self, in_channels: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)
