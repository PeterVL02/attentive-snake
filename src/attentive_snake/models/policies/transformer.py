"""Actor-critic transformer over the observation's positions (``kind = "transformer"``).

Children of ``self.net``, in order:
embed -> positional -> cls -> blocks (each applied in turn) -> norm -> pool -> head (policy_head + value_head, or q_head: models.heads)

Grid: every cell is one token built from all its channels. Vector: every element (number)
is one token, a value embedding plus a learned element embedding, so attention is over
the observation's features. With history (K > 1) the stacked frames simply appear as
extra values of each token. ``embed`` / ``positional`` come from ``TOKEN_STAGES``.
Token order: [class token, position 0 .. position P-1] (row-major cells, or elements in
order); see ``token_layout``.

Pooling uses a learned class token rather than mean pooling: the snake head is a
single cell out of H*W, and a mean would dilute it to 1/(H*W) of the signal, whereas
the class token can attend to the positions that matter (head, nearby body, food).
"""

from torch import Tensor, nn

from attentive_snake.models.input_stages.catalog import TOKEN_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import run_stage, whole_observation
from attentive_snake.models.policies.base import AttentivePolicy
from attentive_snake.models.token_layout import TokenLayout


class SnakeTransformerPolicy(AttentivePolicy):
    observation_kinds = readable_kinds(TOKEN_STAGES)

    def _build_front(self) -> dict[str, nn.Module]:
        stage = self._input_stage(TOKEN_STAGES, whole_observation(self.observation_spec))
        self._stage_names, self._positions = tuple(stage.modules), stage.num_tokens
        return dict(stage.modules)

    def _tokens(self, obs: Tensor) -> Tensor:
        return run_stage(self.net, self._stage_names, obs)

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(num_frames=1, tokens_per_frame=self._positions)
