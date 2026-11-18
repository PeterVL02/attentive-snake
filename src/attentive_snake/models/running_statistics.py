"""Running mean and variance of a stream of batches, as a module so its numbers travel with a ``state_dict``."""

import torch
from torch import Tensor, nn

_INITIAL_COUNT = 1e-4  # as Gymnasium's RunningMeanStd: the first batch outweighs the starting mean 0, variance 1 at once


class RunningStatistics(nn.Module):
    """Mean and population variance per element of everything passed to ``update``, in float64.

    ``shape`` is one sample's shape; ``update`` takes a batch ``(B, *shape)``. Nothing changes
    the numbers but ``update`` and ``load_state_dict``: reading them is free of side effects.
    """

    mean: Tensor
    variance: Tensor
    count: Tensor

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(_INITIAL_COUNT, dtype=torch.float64))

    @torch.no_grad()
    def update(self, batch: Tensor) -> None:
        """Merge a batch of samples into the statistics (the order of the batches does not matter)."""
        samples = batch.to(self.mean.dtype)
        added = samples.shape[0]
        batch_mean = samples.mean(dim=0)
        batch_variance = samples.var(dim=0, correction=0)
        delta = batch_mean - self.mean
        total = self.count + added
        spread = self.variance * self.count + batch_variance * added + delta.pow(2) * self.count * added / total
        self.mean.copy_(self.mean + delta * added / total)
        self.variance.copy_(spread / total)
        self.count.copy_(total)
