"""The selectable policy architectures, by ``ModelConfig.kind``."""

from collections.abc import Callable

from torch import nn

from attentive_snake.models.cnn_policy import CnnPolicy
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.frame_attention_policy import FrameAttentionPolicy
from attentive_snake.models.hybrid_policy import HybridPolicy
from attentive_snake.models.kind_info import CONV_FIELDS, TRANSFORMER_FIELDS, KindInfo
from attentive_snake.models.temporal_transformer_policy import TemporalTransformerPolicy
from attentive_snake.models.transformer_policy import SnakeTransformerPolicy

_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "transformer": SnakeTransformerPolicy,
    "cnn": CnnPolicy,
    "hybrid": HybridPolicy,
    "temporal_transformer": TemporalTransformerPolicy,
    "frame_attention": FrameAttentionPolicy,
}

POLICY_KINDS: tuple[str, ...] = tuple(_BUILDERS)

KIND_INFO: dict[str, KindInfo] = {
    "transformer": KindInfo(
        "Every board cell becomes a token and a transformer lets a class token attend to the cells that matter. "
        "Stacked history frames are treated as extra channels of each cell.",
        TRANSFORMER_FIELDS,
        False,
        "grid",
    ),
    "cnn": KindInfo(
        "A plain convolutional network without attention, used as a baseline. "
        "It reads local patterns on the board and then looks at the whole feature map at once.",
        ("d_model", *CONV_FIELDS),
        False,
        None,
    ),
    "hybrid": KindInfo(
        "A convolutional stem summarises each cell's neighbourhood, then a transformer attends over the cells. "
        "Combines local pattern detection with global attention.",
        (*TRANSFORMER_FIELDS, *CONV_FIELDS),
        False,
        "grid",
    ),
    "temporal_transformer": KindInfo(
        "A transformer over the cells of several past frames (plus one token per past action, if enabled), "
        "so attention can look across both space and time.",
        TRANSFORMER_FIELDS,
        True,
        "frames",
    ),
    "frame_attention": KindInfo(
        "A convolutional network encodes each past frame into one vector, and attention decides which frames matter. "
        "The attention shown is one weight per frame.",
        (*TRANSFORMER_FIELDS, *CONV_FIELDS),
        True,
        "weights",
    ),
}


def build_policy(
    config: ModelConfig,
    observation_shape: tuple[int, int, int],
    num_actions: int,
    history_length: int = 1,
) -> nn.Module:
    if config.kind not in _BUILDERS:
        raise ValueError(f"unknown model kind {config.kind!r}; valid kinds: {', '.join(POLICY_KINDS)}")
    return _BUILDERS[config.kind](config, observation_shape, num_actions, history_length)
