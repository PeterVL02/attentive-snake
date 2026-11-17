"""The probability distribution over actions that an actor-critic output describes, for sampling and for PPO's loss.

``PolicyOutput``: a categorical over the action indices (softmax of the logits).
``GaussianPolicyOutput``: independent normals, one per dimension of the action vector, joined so
that ``log_prob`` and ``entropy`` are one number per observation (the sum over the dimensions),
exactly what PPO's ratio and entropy bonus need for either kind.
"""

from torch import Tensor
from torch.distributions import Categorical, Distribution, Independent, Normal

from attentive_snake.models.types import GaussianPolicyOutput, PolicyOutput

type ActorCriticOutput = PolicyOutput | GaussianPolicyOutput


def policy_distribution(output: ActorCriticOutput) -> Distribution:
    """A distribution with batch shape (B,) whose samples are actions: (B,) indices or (B, dimension) vectors."""
    if isinstance(output, GaussianPolicyOutput):
        return Independent(Normal(output.mean, output.log_std.exp()), 1)
    if isinstance(output, PolicyOutput):
        return Categorical(logits=output.logits)
    raise TypeError(f"not an actor-critic output: {type(output).__name__}")


def action_parameters(output: ActorCriticOutput) -> Tensor:
    """(B, num_actions): the logits, or the mean action; its width is the env's ``num_actions`` either way."""
    return output.mean if isinstance(output, GaussianPolicyOutput) else output.logits
