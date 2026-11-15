"""The dueling combination (Wang et al., 2016): a state value and per-action advantages -> Q-values."""

from torch import Tensor


def dueling_q_values(value: Tensor, advantages: Tensor) -> Tensor:
    """``Q = V + A - mean(A)``: ``value`` (B,), ``advantages`` (B, A) -> (B, A).

    Removing the mean advantage makes the split unique: the mean of the Q-values over the
    actions is exactly ``value``, and the advantages only say how the actions differ.
    """
    return value.unsqueeze(-1) + advantages - advantages.mean(dim=-1, keepdim=True)
