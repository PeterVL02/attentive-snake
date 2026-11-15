"""The jepa world model with a per-token predictor on the hybrid trunk (``kind = "jepa_tokens"``).

Children of ``self.net``, in order: the ``hybrid`` kind's up to its final norm (stem -> project ->
[tokens] -> [positional] -> cls -> blocks -> norm; no pool), then ``predictor``
(``blocks.token_predictor.TokenPredictor``), then the jepa head (``models.jepa.heads``).

The state embedding is the whole encoded token set (B, 1 + P, D): the class token and one token
per board cell (per number of a vector observation). The predictor predicts that whole set for
every action, the decoders read its class token (``jepa.token_world_model.TokenWorldModel``).
``num_layers`` is the encoder's transformer blocks, ``predictor_layers`` the predictor's.
Attention maps and the token layout are hybrid's (the encoder's), so heatmaps and diagrams
work as for ``hybrid``.

``_pooled`` returns the token set: it is what the head and ``encode`` read.
"""

from torch import Tensor, nn

from attentive_snake.models.jepa.token_world_model import TokenWorldModel
from attentive_snake.models.policies.base import encode_tokens
from attentive_snake.models.policies.hybrid import HybridPolicy
from attentive_snake.models.policies.token_encoder import TokenSetEncoder


class JepaTokensPolicy(TokenWorldModel, HybridPolicy):
    def _build_net(self) -> dict[str, nn.Module]:
        trunk = super()._build_net()
        del trunk["pool"]  # the decoders read the class token of a token set (TokenWorldModel.readout)
        return {**trunk, **self._predictor_modules()}

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return encode_tokens(self.net, self._tokens(obs), need_weights)

    def encode(self, obs: Tensor) -> Tensor:
        tokens, _ = self._pooled(obs, need_weights=False)
        return tokens

    def embedding_encoder(self) -> nn.Module:
        return TokenSetEncoder([self.net[name] for name in self._stage_names], self.net)
