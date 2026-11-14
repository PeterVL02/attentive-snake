"""The selectable policy architectures, by ``ModelConfig.kind``."""

from collections.abc import Callable

from torch import nn

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.heads import DEFAULT_HEAD
from attentive_snake.models.kind_info import CONV_FIELDS, TRANSFORMER_FIELDS, InputReading, KindInfo
from attentive_snake.models.policies.cnn import CnnPolicy
from attentive_snake.models.policies.frame_attention import FrameAttentionPolicy
from attentive_snake.models.policies.hybrid import HybridPolicy
from attentive_snake.models.policies.jepa import JepaPolicy
from attentive_snake.models.policies.lstm import LstmPolicy
from attentive_snake.models.policies.mlp import MlpPolicy
from attentive_snake.models.policies.resnet import ResnetPolicy
from attentive_snake.models.policies.temporal_transformer import TemporalTransformerPolicy
from attentive_snake.models.policies.transformer import SnakeTransformerPolicy
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training_metric import TrainingMetric

_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "transformer": SnakeTransformerPolicy,
    "cnn": CnnPolicy,
    "resnet": ResnetPolicy,
    "hybrid": HybridPolicy,
    "temporal_transformer": TemporalTransformerPolicy,
    "frame_attention": FrameAttentionPolicy,
    "lstm": LstmPolicy,
    "jepa": JepaPolicy,
    "mlp": MlpPolicy,
}

POLICY_KINDS: tuple[str, ...] = tuple(_BUILDERS)

NO_KERNEL = ("kernel_size",)  # nothing to convolve in a list of numbers

_JEPA_SWITCH = "model.prediction_coef"
JEPA_METRICS = (
    TrainingMetric("prediction_loss", "prediction loss", "jepa: mean cosine distance (0 = same direction, 1 = unrelated, 2 = opposite) between the embedding predicted for the action taken and the target encoder's embedding of the state that followed, over the gradient steps since the previous record.", _JEPA_SWITCH),
    TrainingMetric("embedding_std", "embedding spread", "jepa: how different the encoder's embeddings of a batch of states are (mean per-dimension standard deviation of the unit-length embeddings). Near 0 means collapse: every state looks the same to the model; about 1/sqrt(d_model) means unrelated directions.", _JEPA_SWITCH),
)
_JEPA_HOW = (
    "trained from scratch, no pretrained weights. The predictor then takes the embedding and each action and predicts the embedding of the next state "
    "(num_layers deep), and a value decoder turns embeddings into values: with DQN, Q(s, a) is the value of the embedding action a is predicted to lead to, "
    "so it plays the action whose predicted next state is worth most; with PPO the policy scores the predicted embeddings and the value decoder values the current one. "
    "With prediction_coef > 0 a prediction loss makes the embedding predicted for the action taken match a slowly updated (EMA) copy of the encoder's embedding "
    "of the state that really followed (JEPA-style; prediction_coef = 0 is the same network without it). "
    "value_training (DQN) chooses whether the value decoder learns from the predicted next embedding or from the real one. Experimental."
)

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
    "resnet": KindInfo(
        ("d_model", *CONV_FIELDS),
        False,
        None,
        {
            "grid": InputReading(
                "A residual convolutional network without attention: a convolutional stem, then conv_layers residual blocks "
                "(two convolutions and a skip connection each, same width, no downsampling), then a readout of the whole feature map.",
                True,
            ),
            "vector": InputReading(
                "A residual MLP without attention (as in SimBa): a linear embedding, then conv_layers residual blocks "
                "(layer norm, linear, GELU, linear, plus a skip connection; conv_channels wide) and a linear readout to d_model.",
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
    "lstm": KindInfo(
        ("d_model", "num_layers", *CONV_FIELDS),
        True,
        None,
        {
            "grid": InputReading(
                "A convolutional network encodes each of the last K frames into one vector (plus the past action, if enabled) and an LSTM "
                "(num_layers layers, d_model hidden) reads them oldest to newest. A DRQN-style agent with a fixed unroll length: the memory "
                "is exactly the history window, since the hidden state is not carried from one step to the next or stored in the replay buffer.",
                False,
            ),
            "vector": InputReading(
                "An MLP (conv_channels wide, conv_layers deep) encodes each of the last K frames into one vector (plus the past action, "
                "if enabled) and an LSTM (num_layers layers, d_model hidden) reads them oldest to newest. A DRQN-style agent with a fixed "
                "unroll length: the memory is exactly the history window, since the hidden state is not carried from one step to the "
                "next or stored in the replay buffer. Useful when part of the observation is hidden (Lunar Lander's hide_velocities).",
                False,
                NO_KERNEL,
            ),
        },
    ),
    "jepa": KindInfo(
        ("d_model", "num_layers", *CONV_FIELDS, "prediction_coef", "value_training", "target_ema"),
        False,
        None,
        {
            "grid": InputReading(
                "A world model that acts by predicting where each action leads. A convolutional encoder (as in cnn) turns the board into an embedding, " + _JEPA_HOW,
                False,
            ),
            "vector": InputReading(
                "A world model that acts by predicting where each action leads. An MLP encoder (conv_channels wide, conv_layers deep, as in cnn) "
                "turns the numbers into an embedding, " + _JEPA_HOW,
                False,
                NO_KERNEL,
            ),
        },
        JEPA_METRICS,
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
