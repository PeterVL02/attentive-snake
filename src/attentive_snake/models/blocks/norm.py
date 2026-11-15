"""Optional normalisation layers of the conv / MLP blocks (``ModelConfig.layer_norm``).

A block built without normalisation holds ``None`` where the layer would be: no module is
registered, so its state-dict keys, its diagram and its outputs are those of a network
built before the switch existed.

* Vectors and tokens ``(..., width)``: ``nn.LayerNorm`` over the last axis.
* Feature maps ``(B, C, H, W)``: ``nn.GroupNorm(1, C)``: one group, so every board is
  normalised over all its channels and cells at once (layer normalisation of the whole map,
  with a learned scale and shift per channel). Unlike batch norm it uses no statistics of
  the batch, so collection, replayed batches and the target network all see the same function.
"""

from torch import Tensor, nn


def vector_norm(width: int, enabled: bool) -> nn.Module | None:
    """A layer norm over the last axis of ``(..., width)``, or ``None`` when not ``enabled``."""
    return nn.LayerNorm(width) if enabled else None


def map_norm(channels: int, enabled: bool) -> nn.Module | None:
    """A layer norm over a whole feature map ``(B, channels, H, W)``, or ``None`` when not ``enabled``."""
    return nn.GroupNorm(1, channels) if enabled else None


def normalised(norm: nn.Module | None, x: Tensor) -> Tensor:
    """``norm(x)``, or ``x`` itself where a block has no normalisation."""
    return x if norm is None else norm(x)


def present(*norms: nn.Module | None) -> list[nn.Module]:
    """The normalisation layers that exist, for splicing into an ``nn.Sequential``."""
    return [norm for norm in norms if norm is not None]
