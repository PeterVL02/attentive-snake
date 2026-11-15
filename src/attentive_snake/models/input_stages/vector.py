"""Input stages for vector observations ``(N, values * elements)``: every element (number) is a position.

A list of numbers has no neighbourhoods, so where grids use convolutions these use
fully connected layers (``conv_channels`` wide, ``conv_layers`` deep; ``kernel_size`` is unused).
``config.layer_norm`` adds a layer norm after the plain layers and in the readout
(``blocks.norm``); the residual blocks (``blocks.residual``) are layer-normalised either way.
"""

from torch import nn

from attentive_snake.models.blocks.vector import (
    ElementEmbedding,
    ElementMlpStem,
    ElementPositionalEmbedding,
    LinearReadout,
    MlpEncoder,
)
from attentive_snake.models.blocks.residual import ResidualElementStem, ResidualMlpEncoder
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.input_stages.stage import Stage, StageInput


def _encoder(source: StageInput, config: ModelConfig, residual: bool) -> nn.Module:
    """The MLP over all numbers: plain layers, or an embedding and residual blocks (same shapes)."""
    if residual:
        return ResidualMlpEncoder(source.values * source.channels, config.conv_channels, config.conv_layers)
    return MlpEncoder(source.values * source.channels, config.conv_channels, config.conv_layers, config.layer_norm)


def _element_stem(source: StageInput, config: ModelConfig, residual: bool) -> nn.Module:
    """The per-element MLP making one token per element: plain layers, or residual blocks (same shapes)."""
    if residual:
        return ResidualElementStem(source.values, source.channels, config.conv_channels, config.conv_layers)
    return ElementMlpStem(source.values, source.channels, config.conv_channels, config.conv_layers, config.layer_norm)


def _readout(config: ModelConfig) -> nn.Module:
    """The encoder's features -> one ``d_model`` vector."""
    return LinearReadout(config.conv_channels, config.d_model, config.layer_norm)


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
    return Stage({"encoder": _encoder(source, config, residual=False), "readout": _readout(config)})


def frame_stage(source: StageInput, config: ModelConfig) -> Stage:
    """Like ``feature_stage`` for one frame; residual blocks with ``config.residual``."""
    return Stage({"encoder": _encoder(source, config, config.residual), "readout": _readout(config)})


def hybrid_stage(source: StageInput, config: ModelConfig) -> Stage:
    """A per-element MLP (residual blocks with ``config.residual``) makes one token per element, projected to ``d_model``."""
    return Stage(
        {
            "stem": _element_stem(source, config, config.residual),
            "project": nn.Linear(config.conv_channels, config.d_model),
        },
        num_tokens=source.channels,
    )


def residual_stage(source: StageInput, config: ModelConfig) -> Stage:
    """An embedding and ``conv_layers`` residual MLP blocks over all numbers, then a linear readout to one ``d_model`` vector."""
    return Stage({"encoder": _encoder(source, config, residual=True), "readout": _readout(config)})
