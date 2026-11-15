"""Input stages for image observations ``(N, C, H, W)``: screen pictures, thousands of pixels each.

A token or an unstrided convolution per pixel (what ``grid.py`` does per board cell) would be
far too much, so these stages shrink first: the conv stages with the strided convolutions of
the DQN paper (``blocks.image.DownsamplingConvEncoder``; ``conv_channels`` sets their width,
``conv_layers`` / ``kernel_size`` / ``residual`` have no effect), the token stage by cutting
the picture into ``PATCH_SIZE`` pixel patches. Tokens therefore form a grid coarser than the
picture; the stage says which (``Stage.token_grid``). ``config.layer_norm`` adds a
normalisation after every convolution and in the readout.
"""

from torch import nn

from attentive_snake.models.blocks.conv import FlattenReadout, GridToTokens
from attentive_snake.models.blocks.image import DownsamplingConvEncoder, PatchEmbedding, downsampled_size, patch_count
from attentive_snake.models.blocks.transformer import PositionalEmbedding2D
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.input_stages.stage import Stage, StageInput

PATCH_SIZE = 12  # pixels a side per token of the token stage: 7 x 7 tokens for an 84 x 84 picture


def _conv_encoder(source: StageInput, config: ModelConfig) -> nn.Module:
    return DownsamplingConvEncoder(source.values * source.channels, config.conv_channels, config.layer_norm)


def _feature_map(source: StageInput) -> tuple[int, int]:
    """Rows and columns of the conv encoder's feature map."""
    height, width = source.spatial
    return downsampled_size(height), downsampled_size(width)


def _readout(source: StageInput, config: ModelConfig) -> nn.Module:
    """The whole feature map flattened into one ``d_model`` vector."""
    rows, columns = _feature_map(source)
    return FlattenReadout(config.conv_channels, rows, columns, config.d_model, config.layer_norm)


def token_stage(source: StageInput, config: ModelConfig) -> Stage:
    """One token per ``PATCH_SIZE`` pixel patch from all its channels, plus a learned row + column embedding."""
    height, width = source.spatial
    rows, columns = patch_count(height, PATCH_SIZE), patch_count(width, PATCH_SIZE)
    return Stage(
        {
            "embed": PatchEmbedding(source.values * source.channels, config.d_model, PATCH_SIZE),
            "positional": PositionalEmbedding2D(rows, columns, config.d_model),
        },
        num_tokens=rows * columns,
        token_grid=(rows, columns),
    )


def feature_stage(source: StageInput, config: ModelConfig) -> Stage:
    """Strided convolutions, then the whole feature map flattened into one ``d_model`` vector (the DQN paper's network)."""
    return Stage({"encoder": _conv_encoder(source, config), "readout": _readout(source, config)})


def frame_stage(source: StageInput, config: ModelConfig) -> Stage:
    """``feature_stage`` for one frame."""
    return feature_stage(source, config)


def hybrid_stage(source: StageInput, config: ModelConfig) -> Stage:
    """Strided convolutions summarise the picture into a small feature map, then each of its cells becomes a token."""
    rows, columns = _feature_map(source)
    return Stage(
        {
            "stem": _conv_encoder(source, config),
            "project": nn.Conv2d(config.conv_channels, config.d_model, kernel_size=1),
            "tokens": GridToTokens(),
            "positional": PositionalEmbedding2D(rows, columns, config.d_model),
        },
        num_tokens=rows * columns,
        token_grid=(rows, columns),
    )
