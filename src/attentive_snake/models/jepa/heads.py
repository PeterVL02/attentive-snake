"""The head forms of the jepa kinds: what the encoder's state embedding ``z`` becomes, through the predictor.

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

* ``dueling`` (DQN with ``dqn.dueling``): ``Q(s, a) = V + A(a) - mean(A)`` with ``V = value_decoder(z)``, the
  value of the current embedding, and ``A(a) = advantage_decoder(predictor(z, a))``, a score of the
  embedding each action is predicted to lead to. The value decoder here values the state the
  agent is in, not a predicted next state, so ``value_training = "actual"`` (which trains
  ``value_decoder`` on the real next state as Q(s, a)) does not fit it (``world_model.py`` refuses).

The decoders read one vector per state: a ``readout`` (``models.jepa.readout``) takes it from
the embedding, current or predicted. ``JEPA_HEADS`` read the embedding whole (``jepa``,
``jepa_attention``: ``z`` is (B, D)); ``CLASS_TOKEN_HEADS`` read the class token of a token set
(``jepa_tokens``: ``z`` is (B, T, D), the predictor gives (B, A, T, D)). Either way the
decoders see (B, D) and (B, A, D).
"""

from collections.abc import Callable, Mapping

from torch import Tensor, nn

from attentive_snake.models.blocks.heads import EmbeddingScore, ValueHead
from attentive_snake.models.dueling import dueling_q_values
from attentive_snake.models.heads import POLICY_INIT_GAIN, Q_INIT_GAIN, VALUE_INIT_GAIN, Head, init_head
from attentive_snake.models.jepa.readout import class_token, whole
from attentive_snake.models.types import PolicyOutput, QValues


def _value_decoder(d_model: int) -> ValueHead:
    decoder = ValueHead(d_model)
    init_head(decoder.linear, VALUE_INIT_GAIN)
    return decoder


type Readout = Callable[[Tensor], Tensor]


class PredictedValueHead:
    """``value_decoder`` applied to (the readout of) every action's predicted next embedding -> ``QValues``."""

    def __init__(self, readout: Readout = whole) -> None:
        self._readout = readout

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        return {"value_decoder": _value_decoder(d_model)}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> QValues:
        return QValues(q_values=net["value_decoder"](self._readout(net["predictor"](pooled))))


class PredictedActorCriticHead:
    """``policy_decoder`` scores each action's predicted next embedding; ``value_decoder`` values ``z`` -> ``PolicyOutput``."""

    def __init__(self, readout: Readout = whole) -> None:
        self._readout = readout

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        policy = EmbeddingScore(d_model)
        init_head(policy.linear, POLICY_INIT_GAIN)
        return {"policy_decoder": policy, "value_decoder": _value_decoder(d_model)}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> PolicyOutput:
        read = self._readout
        return PolicyOutput(logits=net["policy_decoder"](read(net["predictor"](pooled))), value=net["value_decoder"](read(pooled)))


class PredictedDuelingHead:
    """``value_decoder`` values ``z``; ``advantage_decoder`` scores each action's predicted next embedding -> ``QValues``."""

    def __init__(self, readout: Readout = whole) -> None:
        self._readout = readout

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        advantage = EmbeddingScore(d_model)
        init_head(advantage.linear, Q_INIT_GAIN)
        return {"value_decoder": _value_decoder(d_model), "advantage_decoder": advantage}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> QValues:
        read = self._readout
        advantages = net["advantage_decoder"](read(net["predictor"](pooled)))
        return QValues(q_values=dueling_q_values(net["value_decoder"](read(pooled)), advantages))


def jepa_heads(readout: Readout) -> dict[str, Head]:
    """The head forms, their decoders reading ``readout`` of the embeddings."""
    return {
        "actor_critic": PredictedActorCriticHead(readout),
        "q_values": PredictedValueHead(readout),
        "dueling": PredictedDuelingHead(readout),
    }


JEPA_HEADS: dict[str, Head] = jepa_heads(whole)
CLASS_TOKEN_HEADS: dict[str, Head] = jepa_heads(class_token)
