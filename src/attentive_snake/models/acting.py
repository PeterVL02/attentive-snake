"""The one place that turns a policy's output (either head) into actions and into the numbers shown next to a game.

``PolicyOutput`` (actor-critic): the greedy action is the most likely one, sampling draws
from the softmax of the logits, the readout is the action probabilities plus the state
value. ``QValues``: the greedy action has the highest Q-value; Q-values define no
distribution, so sampling plays the greedy action too; the readout is the Q-values.
Callers (evaluation, watching) never look at class or algorithm names.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from attentive_snake.models.types import PolicyOutput, QValues

type HeadOutput = PolicyOutput | QValues


@dataclass(frozen=True)
class ActionReadout:
    """What the network says about each action of one observation, for display."""

    kind: str  # "probabilities" | "q_values"
    label: str
    values: list[float]  # one per action
    state_value: float | None  # the value head's estimate (actor-critic only)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "label": self.label, "values": self.values, "state_value": self.state_value}


def action_scores(output: HeadOutput) -> Tensor:
    """(B, num_actions) scores whose argmax is the greedy action: logits or Q-values."""
    if isinstance(output, QValues):
        return output.q_values
    if isinstance(output, PolicyOutput):
        return output.logits
    raise TypeError(f"not a policy output: {type(output).__name__}")


def greedy_actions(output: HeadOutput) -> Tensor:
    """(B,) the best action of each observation."""
    return action_scores(output).argmax(dim=-1)


def can_sample(output: HeadOutput) -> bool:
    """Whether the output defines a distribution over actions (actor-critic) to sample from."""
    return isinstance(output, PolicyOutput)


def sampled_actions(output: HeadOutput) -> Tensor:
    """(B,) actions drawn from the policy's distribution (torch's global RNG); greedy where there is none."""
    if can_sample(output):
        return Categorical(logits=output.logits).sample()
    return greedy_actions(output)


@torch.no_grad()
def action_readout(output: HeadOutput) -> ActionReadout:
    """The readout of the first observation of the batch."""
    if isinstance(output, QValues):
        return ActionReadout("q_values", "Q-values", output.q_values[0].tolist(), None)
    probabilities = torch.softmax(action_scores(output)[0], dim=-1)
    return ActionReadout("probabilities", "action probabilities", probabilities.tolist(), float(output.value[0]))
