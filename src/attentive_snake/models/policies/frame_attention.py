"""One vector per frame, attention across frames (``kind = "frame_attention"``).

Children of ``self.net``, in order:
frame_encoder -> [action_embed] -> temporal -> cls -> blocks -> norm -> pool -> head (policy_head + value_head, or q_head: models.heads)

``frame_encoder`` (shared by all frames) encodes each frame's game channels into one
``d_model`` vector (``FRAME_STAGES``: convolutions + flatten readout for grids, an MLP +
linear readout for vectors). That vector gets the frame's action embedding added (when
action channels exist; a learned "no action" entry for all-zero channels) and a learned
per-frame embedding.

Token order (see ``token_layout``), T = 1 + K:  [ class | frame 0 | ... | frame K-1 ]
Frame 0 is the oldest, frame K-1 the newest.

Readout is the class token rather than the newest frame's token: the same pool as the
other attention policies, and it can mix any frames, with its attention over frames
directly showing which frame the decision draws on (the newest frame can still be
attended by the class token trivially, so nothing is lost).
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.temporal import TemporalEmbedding
from attentive_snake.models.input_stages.catalog import FRAME_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import one_frame
from attentive_snake.models.policies.base import AttentivePolicy
from attentive_snake.models.policies.frame_vectors import frame_vector_modules, frame_vectors
from attentive_snake.models.token_layout import TokenLayout


class FrameAttentionPolicy(AttentivePolicy):
    observation_kinds = readable_kinds(FRAME_STAGES)

    def _build_front(self) -> dict[str, nn.Module]:
        stage = self._input_stage(FRAME_STAGES, one_frame(self.observation_spec))
        front = frame_vector_modules(stage, self.observation_spec, self.num_actions, self.config.d_model)
        front["temporal"] = TemporalEmbedding(self.history_length, self.config.d_model)
        return front

    def _tokens(self, obs: Tensor) -> Tensor:
        return self.net["temporal"](frame_vectors(self.net, obs, self.observation_spec, self.num_actions))

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(num_frames=self.history_length, tokens_per_frame=1)
