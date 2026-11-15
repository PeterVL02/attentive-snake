"""``TokenWorldModel``: the jepa world model on a token set instead of one vector per state.

For a policy whose ``encode`` gives every token of its encoder (B, T, D), class token first
(``jepa_tokens``). It changes three things of ``JepaWorldModel``: the predictor predicts every
token (``blocks.token_predictor.TokenPredictor``, ``predictor_layers`` blocks), the decoders read
the class token (``readout``, ``CLASS_TOKEN_HEADS``), and so does the prediction objective's value
gap. The objective itself compares the token sets token by token (``models.jepa.objective``).
"""

from typing import ClassVar

from torch import Tensor, nn

from attentive_snake.models.blocks.token_predictor import TokenPredictor
from attentive_snake.models.heads import Head
from attentive_snake.models.jepa.heads import CLASS_TOKEN_HEADS
from attentive_snake.models.jepa.readout import class_token
from attentive_snake.models.jepa.world_model import JepaWorldModel


class TokenWorldModel(JepaWorldModel):
    """Mixin: ``JepaWorldModel`` whose embeddings are token sets; the decoders read their class token."""

    heads: ClassVar[dict[str, Head]] = CLASS_TOKEN_HEADS

    def readout(self, embeddings: Tensor) -> Tensor:
        return class_token(embeddings)

    def _predictor_modules(self) -> dict[str, nn.Module]:
        c = self.config
        return {"predictor": TokenPredictor(c.d_model, self.num_actions, c.predictor_layers, c.num_heads, c.mlp_ratio, c.dropout)}
