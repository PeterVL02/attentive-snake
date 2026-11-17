"""The one place that turns a policy's output (any head) into actions and into the numbers shown next to a game.

``PolicyOutput`` (actor-critic, discrete actions): the greedy action is the most likely one,
sampling draws from the softmax of the logits, the readout is the action probabilities plus the
state value. ``GaussianPolicyOutput`` (actor-critic, continuous actions): the greedy action is
the mean, sampling draws from the Gaussian (``models.action_distribution``), the readout is the
mean per dimension with its standard deviation (``spread``) plus the state value; the env clips
either to the game's bounds. ``QValues``: the greedy action has the highest Q-value; Q-values
define no distribution, so sampling plays the greedy action too; the readout is the Q-values.
Callers (evaluation, watching) never look at class or algorithm names.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from attentive_snake.models.action_distribution import policy_distribution
from attentive_snake.models.types import GaussianPolicyOutput, PolicyOutput, QValues

type HeadOutput = PolicyOutput | GaussianPolicyOutput | QValues


@dataclass(frozen=True)
class ActionReadout:
    """What the network says about each action (or each dimension of the action vector) of one observation, for display."""

    kind: str  # "probabilities" | "q_values" | "action_vector"
    label: str
    values: list[float]  # one per action, or per dimension of the action vector
    state_value: float | None  # the value head's estimate (actor-critic only)
    spread: list[float] | None = None  # "action_vector": the standard deviation of each number when sampling

    def to_dict(self) -> dict:
        readout = {"kind": self.kind, "label": self.label, "values": self.values, "state_value": self.state_value}
        return readout if self.spread is None else readout | {"spread": self.spread}


def action_scores(output: PolicyOutput | QValues) -> Tensor:
    """(B, num_actions) scores whose argmax is the greedy action of a discrete head: logits or Q-values."""
    if isinstance(output, QValues):
        return output.q_values
    if isinstance(output, PolicyOutput):
        return output.logits
    raise TypeError(f"not a discrete policy output: {type(output).__name__}")


def greedy_actions(output: HeadOutput) -> Tensor:
    """The best action of each observation: (B,) indices, or (B, dimension) mean actions."""
    if isinstance(output, GaussianPolicyOutput):
        return output.mean
    return action_scores(output).argmax(dim=-1)


def can_sample(output: HeadOutput) -> bool:
    """Whether the output defines a distribution over actions (actor-critic) to sample from."""
    return isinstance(output, (PolicyOutput, GaussianPolicyOutput))


def sampled_actions(output: HeadOutput) -> Tensor:
    """Actions drawn from the policy's distribution (torch's global RNG); greedy where there is none."""
    if can_sample(output):
        return policy_distribution(output).sample()
    return greedy_actions(output)


@torch.no_grad()
def action_readout(output: HeadOutput) -> ActionReadout:
    """The readout of the first observation of the batch."""
    if isinstance(output, QValues):
        return ActionReadout("q_values", "Q-values", output.q_values[0].tolist(), None)
    if isinstance(output, GaussianPolicyOutput):
        spread = output.log_std[0].exp().tolist()
        return ActionReadout("action_vector", "mean action", output.mean[0].tolist(), float(output.value[0]), spread)
    probabilities = torch.softmax(action_scores(output)[0], dim=-1)
    return ActionReadout("probabilities", "action probabilities", probabilities.tolist(), float(output.value[0]))
