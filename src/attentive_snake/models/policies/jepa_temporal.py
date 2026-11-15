"""The jepa world model on the temporal_transformer trunk (``kind = "jepa_temporal"``): ``z`` comes from attention over time.

Children of ``self.net``, in order: the ``temporal_transformer`` kind's (embed -> positional ->
temporal -> [action_embed] -> cls -> blocks -> norm -> pool; for vectors the vector token stage's
names), then ``predictor``, then the jepa head (``models.jepa.heads``: value_decoder, or
policy_decoder + value_decoder).

The class-token pool over every position of the last K frames (plus one token per past action)
is the embedding ``z``, so it can hold what one frame does not show (velocities from positions).
Everything after it is the ``jepa`` kind's, from ``models.jepa.world_model.JepaWorldModel``.
``num_layers`` sets both the number of transformer blocks and the predictor's depth. Attention
maps and the token layout are temporal_transformer's, so heatmaps per frame and diagrams work
as for ``temporal_transformer``.
"""

from torch import Tensor, nn

from attentive_snake.models.jepa.world_model import JepaWorldModel
from attentive_snake.models.policies.temporal_tokens import TemporalTokens
from attentive_snake.models.policies.temporal_transformer import TemporalTransformerPolicy
from attentive_snake.models.policies.token_encoder import TokenEncoder


class JepaTemporalPolicy(JepaWorldModel, TemporalTransformerPolicy):
    def _build_net(self) -> dict[str, nn.Module]:
        return {**super()._build_net(), **self._predictor_modules()}

    def encode(self, obs: Tensor) -> Tensor:
        pooled, _ = self._pooled(obs, need_weights=False)
        return pooled

    def embedding_encoder(self) -> nn.Module:
        tokens = TemporalTokens(self.net, self._stage_names, self.observation_spec, self.num_actions)
        return TokenEncoder([tokens], self.net)
