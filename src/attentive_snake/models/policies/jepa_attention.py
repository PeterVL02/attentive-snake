"""The jepa world model on the hybrid trunk (``kind = "jepa_attention"``): ``z`` comes from attention.

Children of ``self.net``, in order: the ``hybrid`` kind's (stem -> project -> [tokens] ->
[positional] -> cls -> blocks -> norm -> pool), then ``predictor``, then the jepa head
(``models.jepa.heads``: value_decoder, or policy_decoder + value_decoder).

The class-token pool is the embedding ``z``; everything after it (predictor, head forms,
prediction objective, trained values) is the ``jepa`` kind's, from
``models.jepa.world_model.JepaWorldModel``. ``num_layers`` sets both the number of
transformer blocks and the predictor's depth. Attention maps and the token layout are
hybrid's, so heatmaps and diagrams work as for ``hybrid``.
"""

from torch import Tensor, nn

from attentive_snake.models.jepa.world_model import JepaWorldModel
from attentive_snake.models.policies.hybrid import HybridPolicy
from attentive_snake.models.policies.token_encoder import TokenEncoder


class JepaAttentionPolicy(JepaWorldModel, HybridPolicy):
    def _build_net(self) -> dict[str, nn.Module]:
        return {**super()._build_net(), **self._predictor_modules()}

    def encode(self, obs: Tensor) -> Tensor:
        pooled, _ = self._pooled(obs, need_weights=False)
        return pooled

    def embedding_encoder(self) -> nn.Module:
        return TokenEncoder([self.net[name] for name in self._stage_names], self.net)
