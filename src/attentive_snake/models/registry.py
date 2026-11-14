"""The selectable policy architectures, by ``ModelConfig.kind``."""

from collections.abc import Callable

from torch import nn

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.heads import DEFAULT_HEAD
from attentive_snake.models.kind_info import CONV_FIELDS, TRANSFORMER_FIELDS, InputReading, KindInfo
from attentive_snake.models.policies.cnn import CnnPolicy
from attentive_snake.models.policies.frame_attention import FrameAttentionPolicy
from attentive_snake.models.policies.hybrid import HybridPolicy
from attentive_snake.models.policies.mlp import MlpPolicy
from attentive_snake.models.policies.temporal_transformer import TemporalTransformerPolicy
from attentive_snake.models.policies.transformer import SnakeTransformerPolicy
from attentive_snake.observation_spec import ObservationSpec

_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "transformer": SnakeTransformerPolicy,
    "cnn": CnnPolicy,
    "hybrid": HybridPolicy,
    "temporal_transformer": TemporalTransformerPolicy,
    "frame_attention": FrameAttentionPolicy,
    "mlp": MlpPolicy,
}

POLICY_KINDS: tuple[str, ...] = tuple(_BUILDERS)

NO_KERNEL = ("kernel_size",)  # nothing to convolve in a list of numbers

KIND_INFO: dict[str, KindInfo] = {
    "transformer": KindInfo(
        TRANSFORMER_FIELDS,
        False,
        "grid",
        {
            "grid": InputReading(
                "Every board cell becomes a token and a transformer lets a class token attend to the cells that matter. "
                "Stacked history frames are treated as extra channels of each cell.",
                True,
            ),
            "vector": InputReading(
                "Every number of the observation becomes a token (its value plus a learned embedding of which number it is), "
                "and a class token attends to the numbers that matter. Stacked history frames are extra values of each token. "
                "Shows which inputs the decision draws on; usually learns slower than mlp or cnn.",
                False,
            ),
        },
    ),
    "cnn": KindInfo(
        ("d_model", *CONV_FIELDS),
        False,
        None,
        {
            "grid": InputReading(
                "A plain convolutional network without attention, used as a baseline. "
                "It reads local patterns on the board and then looks at the whole feature map at once.",
                True,
            ),
            "vector": InputReading(
                "A convolution over an arbitrary order of numbers is meaningless, so for these observations this is an MLP: "
                "fully connected layers (conv_channels wide, conv_layers deep) and a linear readout to d_model. "
                "Use the mlp kind for a plain baseline; this one is kept so every kind can play every game.",
                False,
                NO_KERNEL,
            ),
        },
    ),
    "hybrid": KindInfo(
        (*TRANSFORMER_FIELDS, *CONV_FIELDS),
        False,
        "grid",
        {
            "grid": InputReading(
                "A convolutional stem summarises each cell's neighbourhood, then a transformer attends over the cells. "
                "Combines local pattern detection with global attention.",
                True,
            ),
            "vector": InputReading(
                "A small MLP turns every number of the observation into a token (conv_channels wide, conv_layers deep, "
                "shared by all numbers), then a transformer attends over these tokens.",
                False,
                NO_KERNEL,
            ),
        },
    ),
    "temporal_transformer": KindInfo(
        TRANSFORMER_FIELDS,
        True,
        "frames",
        {
            "grid": InputReading(
                "A transformer over the cells of several past frames (plus one token per past action, if enabled), "
                "so attention can look across both space and time.",
                True,
            ),
            "vector": InputReading(
                "A transformer over every number of several past frames (plus one token per past action, if enabled), "
                "so attention can look across both the observation's features and time.",
                False,
            ),
        },
    ),
    "frame_attention": KindInfo(
        (*TRANSFORMER_FIELDS, *CONV_FIELDS),
        True,
        "weights",
        {
            "grid": InputReading(
                "A convolutional network encodes each past frame into one vector, and attention decides which frames matter. "
                "The attention shown is one weight per frame.",
                True,
            ),
            "vector": InputReading(
                "An MLP (conv_channels wide, conv_layers deep) encodes each past frame's numbers into one vector, "
                "and attention decides which frames matter. The attention shown is one weight per frame.",
                False,
                NO_KERNEL,
            ),
        },
    ),
    "mlp": KindInfo(
        ("d_model", "num_layers"),
        False,
        None,
        {
            "grid": InputReading(
                "A plain multilayer perceptron on the flattened board (d_model wide, num_layers deep). "
                "It knows nothing about cells or neighbours, so it is a weak baseline for boards.",
                False,
            ),
            "vector": InputReading(
                "A plain multilayer perceptron on all numbers of the observation (d_model wide, num_layers deep), "
                "without attention: the standard baseline for games described by a handful of numbers.",
                True,
            ),
        },
    ),
}


def build_policy(
    config: ModelConfig, observation_spec: ObservationSpec, num_actions: int, head: str = DEFAULT_HEAD
) -> nn.Module:
    """The policy of ``config.kind`` for observations laid out as ``observation_spec``, ending in ``head`` (``models.heads``)."""
    if config.kind not in _BUILDERS:
        raise ValueError(f"unknown model kind {config.kind!r}; valid kinds: {', '.join(POLICY_KINDS)}")
    return _BUILDERS[config.kind](config, observation_spec, num_actions, head)
