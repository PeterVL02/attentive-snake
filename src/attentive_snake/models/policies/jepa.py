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
and the same network learns from the algorithm's loss alone. No attention. Everything
jepa-specific comes from ``models.jepa.world_model.JepaWorldModel``; this module only says
how an observation becomes ``z``.
"""

from torch import Tensor, nn

from attentive_snake.models.input_stages.catalog import FEATURE_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import run_stage, whole_observation
from attentive_snake.models.jepa.world_model import JepaWorldModel
from attentive_snake.models.policies.base import BasePolicy


class JepaPolicy(JepaWorldModel, BasePolicy):
    observation_kinds = readable_kinds(FEATURE_STAGES)

    def _build_net(self) -> dict[str, nn.Module]:
        stage = self._input_stage(FEATURE_STAGES, whole_observation(self.observation_spec))
        self._stage_names = tuple(stage.modules)
        return {**stage.modules, **self._predictor_modules()}

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return self.encode(obs), []

    def encode(self, obs: Tensor) -> Tensor:
        return run_stage(self.net, self._stage_names, obs)

    def embedding_encoder(self) -> nn.Module:
        return nn.Sequential(*(self.net[name] for name in self._stage_names))
