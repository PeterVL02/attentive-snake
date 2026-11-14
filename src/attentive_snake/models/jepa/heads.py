"""The two head forms of the ``jepa`` kind: what its encoder embedding ``z`` becomes, through the predictor.

Both use the trunk's ``predictor`` (``net["predictor"]``: ``z`` -> one predicted next-state
embedding per action) and add their decoders to ``net``:

* ``q_values`` (DQN): ``Q(s, a) = value_decoder(predictor(z, a))`` - the value of the
  embedding each action is predicted to lead to; the greedy action is the one whose predicted
  next state is worth most.
* ``actor_critic`` (PPO): ``logits[a] = policy_decoder(predictor(z, a))``, a learned score of
  each predicted next-state embedding, and ``value = value_decoder(z)``, the value of the
  current embedding. The actor gets its own decoder (initialised near zero, so the first
  policy is near uniform) instead of the value decoder: logits are not values (their scale
  is the policy's temperature), and the policy gradient and entropy bonus would otherwise
  pull on the very decoder the critic is trained with.
"""

from collections.abc import Mapping

from torch import Tensor, nn

from attentive_snake.models.blocks.heads import EmbeddingScore, ValueHead
from attentive_snake.models.heads import POLICY_INIT_GAIN, VALUE_INIT_GAIN, Head, init_head
from attentive_snake.models.types import PolicyOutput, QValues


def _value_decoder(d_model: int) -> ValueHead:
    decoder = ValueHead(d_model)
    init_head(decoder.linear, VALUE_INIT_GAIN)
    return decoder


class PredictedValueHead:
    """``value_decoder`` applied to every action's predicted next embedding -> ``QValues``."""

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        return {"value_decoder": _value_decoder(d_model)}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> QValues:
        return QValues(q_values=net["value_decoder"](net["predictor"](pooled)))


class PredictedActorCriticHead:
    """``policy_decoder`` scores each action's predicted next embedding; ``value_decoder`` values ``z`` -> ``PolicyOutput``."""

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        policy = EmbeddingScore(d_model)
        init_head(policy.linear, POLICY_INIT_GAIN)
        return {"policy_decoder": policy, "value_decoder": _value_decoder(d_model)}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> PolicyOutput:
        return PolicyOutput(logits=net["policy_decoder"](net["predictor"](pooled)), value=net["value_decoder"](pooled))


JEPA_HEADS: dict[str, Head] = {"actor_critic": PredictedActorCriticHead(), "q_values": PredictedValueHead()}
