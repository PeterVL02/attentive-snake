"""Architecture hyperparameters of the policy networks.

Input size, action count and history length come from the environment, not from here.
``kind`` selects the architecture (see ``attentive_snake.models.registry``); the
transformer fields apply to every kind with attention, the ``conv_*`` / ``kernel_size``
fields to every kind with a convolutional part (cnn, hybrid, frame_attention).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Network size and shape; which fields a kind uses is declared in ``models.registry.KIND_INFO``."""

    kind: str = field(default="transformer", metadata={"help": "Which network architecture to train; see the model kinds for what each one is."})
    d_model: int = field(default=64, metadata={"help": "Width of the network's internal feature vectors (the embedding size). Bigger can represent more but is slower and easier to overfit."})
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads: parallel 'looks' at the board that each focus on different relations. Must divide d_model."})
    num_layers: int = field(default=3, metadata={"help": "Number of stacked transformer blocks. Deeper models can combine information over more steps of reasoning."})
    mlp_ratio: int = field(default=4, metadata={"help": "Hidden-layer width of each transformer block's feed-forward part, as a multiple of d_model."})
    dropout: float = field(default=0.0, metadata={"help": "Probability of randomly zeroing activations during training to reduce overfitting. RL usually works best with 0."})
    conv_channels: int = field(default=64, metadata={"help": "Number of feature maps in each convolutional layer. More channels detect more kinds of local patterns."})
    conv_layers: int = field(default=3, metadata={"help": "Number of convolutional layers; each one widens the area of the board a cell can 'see'."})
    kernel_size: int = field(default=3, metadata={"help": "Width of the square window of each convolution. Must be odd; 3 looks at a cell and its neighbours."})

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
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
