"""A JEPA-style world-model policy (``kind = "jepa"``): it acts by valuing the next-state embeddings it predicts.

Children of ``self.net``, in order:
encoder -> readout -> predictor -> head (``models.jepa.heads``: value_decoder, or
policy_decoder + value_decoder)

* Encoder (``encoder`` + ``readout``, from ``FEATURE_STAGES``, as in the ``cnn`` kind):
  observation -> embedding ``z`` (``d_model``). Convolutions for grids, an MLP for vectors;
  stacked history frames are plain input channels / numbers.
* ``predictor`` (``blocks.predictor.ActionPredictor``, ``num_layers`` deep): ``z`` and an
  action -> the predicted embedding of the next state, for every action in one pass.
  It reads embeddings only, so it can be applied to its own predictions (lookahead).
* Head: DQN's ``Q(s, a) = value_decoder(predictor(z, a))``; PPO's logits score each
  predicted embedding and its value is ``value_decoder(z)`` (see ``models.jepa.heads``).
  What DQN's TD loss trains is ``value_training`` (``trained_values``): Q(s, a) itself
  (``"predicted"``) or the value of the real next state's embedding (``"actual"``).

Trained from scratch on the game, like every kind. Its auxiliary objective
(``auxiliary_objective``, ``models.jepa.objective``) makes the predictor match the embedding
an EMA target encoder gives the real next state; with ``prediction_coef = 0`` there is none,
and the same network learns from the algorithm's loss alone. No attention.
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.predictor import ActionPredictor
from attentive_snake.models.input_stages.catalog import FEATURE_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import run_stage, whole_observation
from attentive_snake.models.jepa.heads import JEPA_HEADS
from attentive_snake.models.jepa.objective import DISTANCES, PredictionObjective
from attentive_snake.models.policies.base import BasePolicy


class JepaPolicy(BasePolicy):
    observation_kinds = readable_kinds(FEATURE_STAGES)
    heads = JEPA_HEADS

    def _build_net(self) -> dict[str, nn.Module]:
        c = self.config
        stage = self._input_stage(FEATURE_STAGES, whole_observation(self.observation_spec))
        self._stage_names = tuple(stage.modules)
        return {**stage.modules, "predictor": ActionPredictor(c.d_model, self.num_actions, c.num_layers)}

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return self.encode(obs), []

    def encode(self, obs: Tensor) -> Tensor:
        """Observations -> embeddings ``z`` (B, d_model)."""
        return run_stage(self.net, self._stage_names, obs)

    @property
    def predictor(self) -> ActionPredictor:
        """Embedding (+ action) -> predicted next-state embedding; callable on its own predictions."""
        return self.net["predictor"]

    def auxiliary_objective(self) -> PredictionObjective | None:
        """The prediction loss with a fresh EMA target encoder, or ``None`` when ``prediction_coef`` is 0."""
        if self.config.prediction_coef == 0.0:
            return None
        encoder = nn.Sequential(*(self.net[name] for name in self._stage_names))  # shares the policy's modules
        c = self.config
        return PredictionObjective(encoder, self.predictor, c.prediction_coef, c.target_ema, DISTANCES[c.value_training])

    def trained_values(self, observations: Tensor, actions: Tensor, next_observations: Tensor) -> Tensor | None:
        """What DQN's TD loss trains (``models.value_training``): ``None`` = Q(s, a) as acted on (``"predicted"``),
        else the value of the real next state's embedding, ``value_decoder(encoder(s'))`` (``"actual"``).

        With ``"actual"`` the TD gradient reaches the value decoder and the online encoder (through s'), never the
        predictor: the predictor appears only in the bootstrap target, which carries no gradient.
        """
        if self.config.value_training == "predicted":
            return None
        return self.net["value_decoder"](self.encode(next_observations))
