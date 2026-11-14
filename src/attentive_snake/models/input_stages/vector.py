"""Input stages for vector observations ``(N, values * elements)``: every element (number) is a position.

A list of numbers has no neighbourhoods, so where grids use convolutions these use
fully connected layers (``conv_channels`` wide, ``conv_layers`` deep; ``kernel_size`` is unused).
"""

from torch import nn

from attentive_snake.models.blocks.vector import (
    ElementEmbedding,
    ElementMlpStem,
    ElementPositionalEmbedding,
    LinearReadout,
    MlpEncoder,
)
from attentive_snake.models.blocks.residual import ResidualMlpEncoder
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.input_stages.stage import Stage, StageInput


def token_stage(source: StageInput, config: ModelConfig) -> Stage:
    """One token per element: a value embedding of its numbers plus a learned element embedding."""
    return Stage(
        {
            "embed": ElementEmbedding(source.values, config.d_model),
            "positional": ElementPositionalEmbedding(source.channels, config.d_model),
        },
        num_tokens=source.channels,
    )


def feature_stage(source: StageInput, config: ModelConfig) -> Stage:
    """An MLP over all numbers, then a linear readout to one ``d_model`` vector."""
    return Stage(
        {
            "encoder": MlpEncoder(source.values * source.channels, config.conv_channels, config.conv_layers),
            "readout": LinearReadout(config.conv_channels, config.d_model),
        }
    )


def hybrid_stage(source: StageInput, config: ModelConfig) -> Stage:
    """A per-element MLP makes one token per element, projected to ``d_model``."""
    return Stage(
        {
            "stem": ElementMlpStem(source.values, source.channels, config.conv_channels, config.conv_layers),
            "project": nn.Linear(config.conv_channels, config.d_model),
        },
        num_tokens=source.channels,
    )


def residual_stage(source: StageInput, config: ModelConfig) -> Stage:
    """An embedding and ``conv_layers`` residual MLP blocks over all numbers, then a linear readout to one ``d_model`` vector."""
    return Stage(
        {
            "encoder": ResidualMlpEncoder(source.values * source.channels, config.conv_channels, config.conv_layers),
            "readout": LinearReadout(config.conv_channels, config.d_model),
        }
    )
