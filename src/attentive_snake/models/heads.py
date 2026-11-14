"""Policy heads: what a policy's pooled features become, chosen by name (the training algorithm names it, never ``ModelConfig``).

A head contributes named, initialised modules to the policy's ``self.net`` (so the
architecture diagram shows them) and turns pooled features into the output contract of
``models.types``. The trunk (everything before the head) is the same for every head.
Adding one (e.g. a dueling head) = a class here and an entry in ``HEADS``.

A kind whose output is not a linear readout of one pooled vector (``jepa``: values of
predicted next-state embeddings) keeps its own table with the same names
(``BasePolicy.heads``); its heads may use the kind's trunk modules in ``net`` by name.
"""

from collections.abc import Mapping
from typing import Protocol

from torch import Tensor, nn

from attentive_snake.models.blocks.heads import PolicyHead, QHead, ValueHead
from attentive_snake.models.types import PolicyOutput, QValues

POLICY_INIT_GAIN = 0.01  # near-uniform initial policy
VALUE_INIT_GAIN = 1.0
Q_INIT_GAIN = 1.0


def init_head(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)


class Head(Protocol):
    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        """The head's modules, initialised, keyed by their name in ``self.net`` (state-dict keys: keep them stable)."""
        ...

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> PolicyOutput | QValues:
        """Pooled features (B, d_model) -> the head's output, using its modules in ``net``."""
        ...


class ActorCriticHead:
    """``policy_head`` (action logits) and ``value_head`` (state value) -> ``PolicyOutput``; trained by PPO."""

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        policy, value = PolicyHead(d_model, num_actions), ValueHead(d_model)
        init_head(policy.linear, POLICY_INIT_GAIN)
        init_head(value.linear, VALUE_INIT_GAIN)
        return {"policy_head": policy, "value_head": value}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> PolicyOutput:
        return PolicyOutput(logits=net["policy_head"](pooled), value=net["value_head"](pooled))


class QValueHead:
    """``q_head``: one Q-value per action -> ``QValues``; trained by DQN."""

    def modules(self, d_model: int, num_actions: int) -> dict[str, nn.Module]:
        q = QHead(d_model, num_actions)
        init_head(q.linear, Q_INIT_GAIN)
        return {"q_head": q}

    def output(self, net: Mapping[str, nn.Module], pooled: Tensor) -> QValues:
        return QValues(q_values=net["q_head"](pooled))


HEADS: dict[str, Head] = {"actor_critic": ActorCriticHead(), "q_values": QValueHead()}
DEFAULT_HEAD = "actor_critic"


def get_head(name: str, table: Mapping[str, Head] = HEADS) -> Head:
    """The head called ``name`` in ``table`` (a kind with its own head forms, e.g. ``jepa``, passes its table; same names)."""
    if name not in table:
        raise ValueError(f"unknown policy head {name!r}; valid heads: {', '.join(table)}")
    return table[name]
