"""``JepaWorldModel``: what turns a policy that embeds observations into a jepa kind.

Shared by every jepa kind (``jepa``: a cnn encoder; ``jepa_attention``: the hybrid trunk). It
assumes one thing of the policy it is mixed into (before ``BasePolicy`` in the bases): it can
turn observations into embeddings ``z`` (B, d_model), through ``encode`` and as one module
(``embedding_encoder``, for the EMA target copy). On top of that it adds

* the ``predictor`` trunk module (``_predictor_modules``; ``z`` + action -> predicted next ``z``),
* the head forms ``JEPA_HEADS`` (values / scores of the predicted embeddings),
* the prediction objective (``auxiliary_objective``, ``None`` with ``prediction_coef = 0``),
* what DQN's TD loss trains (``trained_values``, ``models.value_training``).
"""

from typing import ClassVar

from torch import Tensor, nn

from attentive_snake.models.blocks.predictor import ActionPredictor
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.heads import Head
from attentive_snake.models.jepa.heads import JEPA_HEADS
from attentive_snake.models.jepa.objective import DISTANCES, PredictionObjective


class JepaWorldModel:
    """Mixin: predictor, head forms, prediction objective and trained values for a policy that has ``encode``."""

    heads: ClassVar[dict[str, Head]] = JEPA_HEADS
    config: ModelConfig
    num_actions: int
    net: nn.ModuleDict

    def encode(self, obs: Tensor) -> Tensor:
        """Observations -> embeddings ``z`` (B, d_model)."""
        raise NotImplementedError

    def embedding_encoder(self) -> nn.Module:
        """The observation -> ``z`` path as one module sharing the policy's modules (not registered on the policy)."""
        raise NotImplementedError

    def _predictor_modules(self) -> dict[str, nn.Module]:
        """The trunk's ``predictor`` (``num_layers`` deep), to follow the encoder in ``net``."""
        return {"predictor": ActionPredictor(self.config.d_model, self.num_actions, self.config.num_layers)}

    @property
    def predictor(self) -> ActionPredictor:
        """Embedding (+ action) -> predicted next-state embedding; callable on its own predictions."""
        return self.net["predictor"]

    def auxiliary_objective(self) -> PredictionObjective | None:
        """The prediction loss with a fresh EMA target encoder, or ``None`` when ``prediction_coef`` is 0."""
        c = self.config
        if c.prediction_coef == 0.0:
            return None
        return PredictionObjective(
            self.embedding_encoder(),
            self.predictor,
            c.prediction_coef,
            c.target_ema,
            DISTANCES[c.value_training],
            trains_encoder=c.prediction_trains_encoder,
            value_decoder=self.net["value_decoder"],
            consistent_values=c.value_training == "actual",
        )

    def trained_values(self, observations: Tensor, actions: Tensor, next_observations: Tensor) -> Tensor | None:
        """What DQN's TD loss trains (``models.value_training``): ``None`` = Q(s, a) as acted on (``"predicted"``),
        else the value of the real next state's embedding, ``value_decoder(encode(s'))`` (``"actual"``).

        With ``"actual"`` the TD gradient reaches the value decoder and the online encoder (through s'), never the
        predictor: the predictor appears only in the bootstrap target, which carries no gradient.
        """
        if self.config.value_training == "predicted":
            return None
        return self.net["value_decoder"](self.encode(next_observations))
