"""Architecture hyperparameters of the policy networks.

Input size, action count and history length come from the environment, not from here.
``kind`` selects the architecture (see ``attentive_snake.models.registry``); the
transformer fields apply to every kind with attention, the ``conv_*`` / ``kernel_size``
fields to every kind with a convolutional part (cnn, resnet, hybrid, frame_attention, lstm; for vector
observations these parts are fully connected layers of the same width and depth and
``kernel_size`` is unused); the ``mlp`` kind uses ``d_model`` and ``num_layers`` only; the ``lstm`` kind also uses ``num_layers`` (LSTM layers).
The ``jepa`` kind uses the ``conv_*`` fields for its encoder, ``num_layers`` for its predictor, and
``prediction_coef`` / ``target_ema`` for its prediction loss and ``value_training`` for what DQN's
TD loss trains. Those three do not shape the network,
only how it is trained (``OBJECTIVE_FIELDS``), so a run continued from a checkpoint may change them.
"""

from dataclasses import dataclass, field

OBJECTIVE_FIELDS = ("prediction_coef", "target_ema", "value_training")  # shape the training objective, not the network
VALUE_TRAINING = ("predicted", "actual")  # choices of ModelConfig.value_training


@dataclass(frozen=True)
class ModelConfig:
    """Network size and shape; which fields a kind uses is declared in ``models.registry.KIND_INFO``."""

    kind: str = field(default="transformer", metadata={"help": "Which network architecture to train; see the model kinds for what each one is."})
    d_model: int = field(default=64, metadata={"help": "Width of the network's internal feature vectors (the embedding size; for the mlp kind the layer width). Bigger can represent more but is slower and easier to overfit."})
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads: parallel 'looks' at the board that each focus on different relations. Must divide d_model."})
    num_layers: int = field(default=3, metadata={"help": "Number of stacked transformer blocks (for the mlp kind: fully connected layers; for the lstm kind: LSTM layers). Deeper models can combine information over more steps of reasoning."})
    mlp_ratio: int = field(default=4, metadata={"help": "Hidden-layer width of each transformer block's feed-forward part, as a multiple of d_model."})
    dropout: float = field(default=0.0, metadata={"help": "Probability of randomly zeroing activations during training to reduce overfitting. RL usually works best with 0."})
    conv_channels: int = field(default=64, metadata={"help": "Number of feature maps in each convolutional layer (for vector observations: width of the fully connected layers that replace the convolutions). More channels detect more kinds of local patterns."})
    conv_layers: int = field(default=3, metadata={"help": "Number of convolutional layers; each one widens the area of the board a cell can 'see' (for vector observations: number of fully connected layers that replace them; for the resnet kind: number of residual blocks)."})
    kernel_size: int = field(default=3, metadata={"help": "Width of the square window of each convolution. Must be odd; 3 looks at a cell and its neighbours. Unused for vector observations."})
    prediction_coef: float = field(default=1.0, metadata={"help": "jepa only: weight of the prediction loss, which trains the encoder and predictor so that the embedding predicted for the action taken matches the target encoder's embedding of the state that really followed. It is added to the algorithm's own loss. 0 switches it off (the same network, trained by the reinforcement-learning loss alone: the ablation)."})
    value_training: str = field(default="predicted", metadata={"help": "jepa with DQN only: which embedding the value decoder learns from. Either way the model acts the same, by decoding the predicted next-state embedding of every action and picking the best. 'predicted': the TD loss is applied to the value of the embedding predicted for the action taken, so it also trains the predictor and the encoder (the prediction loss is a helper). 'actual': the TD loss is applied to the value of the embedding of the next state that really happened (the value of arriving there, the reward for getting there included); the predictor learns only from the prediction loss, which then matches the whole embedding, not just its direction, so prediction_coef must be above 0. PPO ignores it: its value is always that of the actual current state."})
    target_ema: float = field(default=0.99, metadata={"help": "jepa only: how slowly the target encoder (which embeds the real next states the predictor must match) follows the trained encoder. After every gradient step each target weight becomes target_ema x itself + (1 - target_ema) x the trained weight. Closer to 1 is a steadier but staler target; 0 copies the encoder every step."})

    def __post_init__(self) -> None:
        # Imported here: the registry imports the policies, which import this module.
        from attentive_snake.models.registry import POLICY_KINDS

        if self.kind not in POLICY_KINDS:
            raise ValueError(f"unknown model kind {self.kind!r}; valid kinds: {', '.join(POLICY_KINDS)}")
        for name in (
            "d_model",
            "num_heads",
            "num_layers",
            "mlp_ratio",
            "conv_channels",
            "conv_layers",
            "kernel_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd (same-padded convolutions), got {self.kernel_size}")
        if self.prediction_coef < 0.0:
            raise ValueError(f"prediction_coef must be >= 0, got {self.prediction_coef}")
        if self.value_training not in VALUE_TRAINING:
            raise ValueError(f"value_training must be one of {', '.join(VALUE_TRAINING)}, got {self.value_training!r}")
        if self.value_training == "actual" and self.prediction_coef == 0.0:
            raise ValueError("value_training = 'actual' needs prediction_coef > 0: only the prediction loss trains the predictor then.")
        if not 0.0 <= self.target_ema <= 1.0:
            raise ValueError(f"target_ema must be in [0, 1], got {self.target_ema}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
