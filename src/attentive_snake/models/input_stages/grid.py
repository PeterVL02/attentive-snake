"""Input stages for grid observations ``(N, C, H, W)``: every cell is a position.

The module names here are the checkpoint keys of every Snake model trained so far.
"""

from torch import nn

from attentive_snake.models.blocks.conv import ConvEncoder, FlattenReadout, GridToTokens
from attentive_snake.models.blocks.residual import ResidualConvEncoder
from attentive_snake.models.blocks.transformer import CellEmbedding, PositionalEmbedding2D
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.input_stages.stage import Stage, StageInput


def _conv_encoder(source: StageInput, config: ModelConfig, residual: bool) -> nn.Module:
    """The convolutions over the board: plain layers, or a widening layer and residual blocks (same shapes)."""
    encoder = ResidualConvEncoder if residual else ConvEncoder
    return encoder(source.values * source.channels, config.conv_channels, config.conv_layers, config.kernel_size)


def token_stage(source: StageInput, config: ModelConfig) -> Stage:
    """One token per cell from all its channels, plus a learned row + column embedding."""
    height, width = source.spatial
    return Stage(
        {
            "embed": CellEmbedding(source.values * source.channels, config.d_model),
            "positional": PositionalEmbedding2D(height, width, config.d_model),
        },
        num_tokens=height * width,
    )


def feature_stage(source: StageInput, config: ModelConfig) -> Stage:
    """Convolutions, then the whole feature map flattened into one ``d_model`` vector."""
    height, width = source.spatial
    return Stage(
        {
            "encoder": _conv_encoder(source, config, residual=False),
            "readout": FlattenReadout(config.conv_channels, height, width, config.d_model),
        }
    )


def frame_stage(source: StageInput, config: ModelConfig) -> Stage:
    """Like ``feature_stage`` for one frame (its modules are named ``conv``/``readout``, as in old checkpoints); residual blocks with ``config.residual``."""
    height, width = source.spatial
    return Stage(
        {
            "conv": _conv_encoder(source, config, config.residual),
            "readout": FlattenReadout(config.conv_channels, height, width, config.d_model),
        }
    )


def hybrid_stage(source: StageInput, config: ModelConfig) -> Stage:
    """Convolutions (residual blocks with ``config.residual``) summarise each cell's neighbourhood, then each cell becomes a token."""
    c = config
    height, width = source.spatial
    return Stage(
        {
            "stem": _conv_encoder(source, config, config.residual),
            "project": nn.Conv2d(c.conv_channels, c.d_model, kernel_size=1),
            "tokens": GridToTokens(),
            "positional": PositionalEmbedding2D(height, width, c.d_model),
        },
        num_tokens=height * width,
    )


def residual_stage(source: StageInput, config: ModelConfig) -> Stage:
    """A conv stem and ``conv_layers`` residual conv blocks, then the feature map flattened into one ``d_model`` vector."""
    height, width = source.spatial
    return Stage(
        {
            "encoder": _conv_encoder(source, config, residual=True),
            "readout": FlattenReadout(config.conv_channels, height, width, config.d_model),
        }
    )
