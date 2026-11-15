"""Architecture hyperparameters of the policy networks.

Input size, action count and history length come from the environment, not from here.
``kind`` selects the architecture (see ``attentive_snake.models.registry``); the
transformer fields apply to every kind with attention, the ``conv_*`` / ``kernel_size``
fields to every kind with a convolutional part (cnn, resnet, hybrid, frame_attention, lstm, jepa, jepa_attention; for vector
observations these parts are fully connected layers of the same width and depth and
``kernel_size`` is unused); ``residual`` (hybrid, jepa_attention, frame_attention, lstm) swaps their plain conv / MLP stem for residual blocks; ``layer_norm`` (every kind with a conv / MLP part: all but transformer, temporal_transformer, jepa_temporal) adds normalisation layers to that part; the ``mlp`` kind uses ``d_model``, ``num_layers`` and ``layer_norm`` only; the ``lstm`` kind also uses ``num_layers`` (LSTM layers).
The ``jepa`` kind uses the ``conv_*`` fields for its encoder, ``num_layers`` for its predictor, and
``prediction_coef`` / ``target_ema`` / ``prediction_trains_encoder`` for its prediction loss and
``value_training`` for what DQN's TD loss trains. ``jepa_attention`` is the same on the hybrid trunk
(transformer and ``conv_*`` fields, ``residual``); its ``num_layers`` sets both the transformer blocks and the predictor.
``jepa_tokens`` is jepa_attention with a per-token predictor: ``num_layers`` sets its encoder's transformer blocks and
``predictor_layers`` (its only reader) the predictor's. ``jepa_temporal`` is jepa on the temporal_transformer trunk
(transformer fields only); its ``num_layers`` sets both the transformer blocks and the predictor. ``prediction_coef`` / ``target_ema`` / ``value_training`` /
``prediction_trains_encoder`` do not shape the network,
only how it is trained (``OBJECTIVE_FIELDS``), so a run continued from a checkpoint may change them.
"""

from dataclasses import dataclass, field

OBJECTIVE_FIELDS = ("prediction_coef", "target_ema", "value_training", "prediction_trains_encoder")  # shape the training objective, not the network
VALUE_TRAINING = ("predicted", "actual")  # choices of ModelConfig.value_training


@dataclass(frozen=True)
class ModelConfig:
    """Network size and shape; which fields a kind uses is declared in ``models.registry.KIND_INFO``."""

    kind: str = field(default="transformer", metadata={"help": "Which network architecture to train; see the model kinds for what each one is."})
    d_model: int = field(default=64, metadata={"help": "Width of the network's internal feature vectors (the embedding size; for the mlp kind the layer width). Bigger can represent more but is slower and easier to overfit."})
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads: parallel 'looks' at the board that each focus on different relations. Must divide d_model."})
    num_layers: int = field(default=3, metadata={"help": "Number of stacked transformer blocks (for the mlp kind: fully connected layers; for the lstm kind: LSTM layers; for jepa: the predictor's layers; for jepa_attention and jepa_temporal: both the transformer blocks and the predictor's layers; for jepa_tokens: the encoder's transformer blocks only, see predictor_layers). Deeper models can combine information over more steps of reasoning."})
    mlp_ratio: int = field(default=4, metadata={"help": "Hidden-layer width of each transformer block's feed-forward part, as a multiple of d_model."})
    dropout: float = field(default=0.0, metadata={"help": "Probability of randomly zeroing activations during training to reduce overfitting. RL usually works best with 0."})
    conv_channels: int = field(default=64, metadata={"help": "Number of feature maps in each convolutional layer (for vector observations: width of the fully connected layers that replace the convolutions). More channels detect more kinds of local patterns."})
    conv_layers: int = field(default=3, metadata={"help": "Number of convolutional layers; each one widens the area of the board a cell can 'see' (for vector observations: number of fully connected layers that replace them; for the resnet kind: number of residual blocks)."})
    kernel_size: int = field(default=3, metadata={"help": "Width of the square window of each convolution. Must be odd; 3 looks at a cell and its neighbours. Unused for vector observations."})
    residual: bool = field(default=False, metadata={"help": "hybrid, jepa_attention, frame_attention and lstm only: build the stem that reads the board (for frame_attention and lstm: the encoder of each frame) from residual blocks (two layers and a skip connection each, conv_layers of them after one widening layer, as in the resnet kind) instead of plain layers. For vector observations the layers are fully connected and the blocks use a layer norm. Other kinds ignore it."})
    layer_norm: bool = field(default=False, metadata={"help": "cnn, resnet, mlp, jepa, hybrid, jepa_attention, jepa_tokens, frame_attention and lstm: add a normalisation layer after every convolution or fully connected layer of the part that reads the board (the conv / MLP trunk, the stem in front of a transformer, the encoder of each frame) and in its readout. A normalisation layer rescales the numbers flowing through the network to a steady size, which keeps long training runs from slowly losing their ability to learn and usually lets them take larger steps safely. On lists of numbers it is a layer norm; on boards it normalises each board's whole feature map at once (a group norm with one group, the layer-norm equivalent for convolutions; it never uses statistics of the batch, unlike batch norm). Off by default: the network is then exactly the one without this switch. The transformer blocks have layer norms of their own either way, as do residual blocks on lists of numbers (resnet on vector games, residual = true), so there only the readout changes; a hybrid-style stem with residual = true on a vector game does not change at all. transformer, temporal_transformer and jepa_temporal ignore it, as do the jepa predictor and the LSTM itself."})
    predictor_layers: int = field(default=1, metadata={"help": "jepa_tokens only: number of transformer blocks of its predictor, which predicts every token (every board cell) of the next state for each action. The predictor runs once per action, so each block costs about as much as one encoder block times the number of actions; one block is usually enough, since a step changes only a few cells. Other kinds ignore it."})
    prediction_coef: float = field(default=0.0, metadata={"help": "jepa kinds only (jepa, jepa_attention, jepa_tokens, jepa_temporal): weight of the prediction loss, which trains the predictor (and, with prediction_trains_encoder, the encoder) so that the embedding predicted for the action taken matches the target encoder's embedding of the state that really followed. It is added to the algorithm's own loss. 0 (the default) leaves it off: the same network, trained by the reinforcement-learning loss alone; measurements so far showed the loss does not beat that and costs 20-35% of the speed. Above 0 it is on (the other jepa objective settings only matter then)."})
    value_training: str = field(default="predicted", metadata={"help": "jepa kinds (jepa, jepa_attention, jepa_tokens, jepa_temporal) with DQN only: which embedding the value decoder learns from. Either way the model acts the same, by decoding the predicted next-state embedding of every action and picking the best. 'predicted' (recommended): the TD loss is applied to the value of the embedding predicted for the action taken, so it trains the decoder exactly where it is used when acting. 'actual' (experimental, did not learn in our tests): the TD loss is applied to the value of the embedding of the next state that really happened (the value of arriving there, the reward for getting there included); the predictor learns only from the prediction loss, which then matches the whole embedding and also the value the decoder gives it, so prediction_coef must be set above 0 (it is 0 by default). Its weakness: one step changes a state so little that the values of different actions differ by less than the predictor's error, so the choice of action is close to a coin toss between the best few. PPO ignores the TD part: its value is always that of the actual current state."})
    prediction_trains_encoder: bool = field(default=False, metadata={"help": "jepa kinds only (jepa, jepa_attention, jepa_tokens, jepa_temporal): whether the prediction loss also trains the encoder. Only matters with prediction_coef above 0. Off (the default): the prediction loss trains the predictor only and the encoder learns from the reinforcement-learning loss alone, as with prediction_coef = 0, so the prediction loss cannot pull the embeddings away from what the values need. On: the encoder also learns to make the next state predictable; in single-seed DQN tests on Lunar Lander that made the agent unstable (it ended up flying off)."})
    target_ema: float = field(default=0.99, metadata={"help": "jepa kinds only (jepa, jepa_attention, jepa_tokens, jepa_temporal): how slowly the target encoder (which embeds the real next states the predictor must match) follows the trained encoder. After every gradient step each target weight becomes target_ema x itself + (1 - target_ema) x the trained weight. Closer to 1 is a steadier but staler target; 0 copies the encoder every step."})

    def __post_init__(self) -> None:
        # Imported here: the registry imports the policies, which import this module.
        from attentive_snake.models.registry import POLICY_KINDS

        if self.kind not in POLICY_KINDS:
            raise ValueError(f"unknown model kind {self.kind!r}; valid kinds: {', '.join(POLICY_KINDS)}")
        for name in (
            "d_model",
            "num_heads",
            "num_layers",
            "predictor_layers",
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
            raise ValueError("value_training = 'actual' needs prediction_coef > 0 (the default 0 leaves the loss off): only the prediction loss trains the predictor then.")
        if not 0.0 <= self.target_ema <= 1.0:
            raise ValueError(f"target_ema must be in [0, 1], got {self.target_ema}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
