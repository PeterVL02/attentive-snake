"""Stem + transformer (``kind = "hybrid"``).

Children of ``self.net``, in order (from ``HYBRID_STAGES``), then the transformer tail:

* grid:   stem -> project -> tokens -> positional -> cls -> blocks -> norm -> pool -> heads.
  ``stem`` is a ConvEncoder, so each cell's feature already summarises its neighbourhood;
  ``project`` (1x1 conv) maps to ``d_model``; ``tokens`` turns cells into tokens.
* vector: stem -> project -> cls -> blocks -> norm -> pool -> heads. ``stem`` is a
  per-element MLP (value + element embedding, residual layers), ``project`` a linear map
  to ``d_model``: still one token per element.

Token order: [class token, position 0 .. position P-1] (row-major cells or elements); see ``token_layout``.
"""

from torch import Tensor, nn

from attentive_snake.models.input_stages.catalog import HYBRID_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import run_stage, whole_observation
from attentive_snake.models.policies.base import AttentivePolicy
from attentive_snake.models.token_layout import TokenLayout


class HybridPolicy(AttentivePolicy):
    observation_kinds = readable_kinds(HYBRID_STAGES)

    def _build_front(self) -> dict[str, nn.Module]:
        stage = self._input_stage(HYBRID_STAGES, whole_observation(self.observation_spec))
        self._stage_names, self._positions = tuple(stage.modules), stage.num_tokens
        return dict(stage.modules)

    def _tokens(self, obs: Tensor) -> Tensor:
        return run_stage(self.net, self._stage_names, obs)

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(num_frames=1, tokens_per_frame=self._positions)
