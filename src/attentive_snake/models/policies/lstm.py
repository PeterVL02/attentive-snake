"""LSTM over the history window (``kind = "lstm"``): a DRQN-style policy with a fixed unroll length.

Children of ``self.net``, in order:
frame_encoder -> [action_embed] -> recurrent -> head (policy_head + value_head, or q_head: models.heads)

``frame_encoder`` (shared by all frames) encodes each frame's game channels into one
``d_model`` vector (``FRAME_STAGES``: convolutions + flatten readout for grids, an MLP +
linear readout for vectors), plus the frame's action embedding when action channels exist.
``recurrent`` is an ``nn.LSTM`` (``num_layers`` layers, hidden size ``d_model``) over the K
frame vectors, oldest to newest; its output after the newest frame is the pooled feature.

Stateless between steps: every call starts from a zero hidden state, so the memory is
exactly the history window (``env.history_length``), as for the temporal kinds. Nothing is
carried across env steps or stored in a replay buffer. No attention, hence no ``attention_maps``.
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.recurrent import LastStateLstm
from attentive_snake.models.input_stages.catalog import FRAME_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import one_frame
from attentive_snake.models.policies.base import BasePolicy
from attentive_snake.models.policies.frame_vectors import frame_vector_modules, frame_vectors


class LstmPolicy(BasePolicy):
    observation_kinds = readable_kinds(FRAME_STAGES)

    def _build_net(self) -> dict[str, nn.Module]:
        c = self.config
        stage = self._input_stage(FRAME_STAGES, one_frame(self.observation_spec))
        return {
            **frame_vector_modules(stage, self.observation_spec, self.num_actions, c.d_model),
            "recurrent": LastStateLstm(c.d_model, c.num_layers),
        }

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        sequence = frame_vectors(self.net, obs, self.observation_spec, self.num_actions)
        return self.net["recurrent"](sequence), []
