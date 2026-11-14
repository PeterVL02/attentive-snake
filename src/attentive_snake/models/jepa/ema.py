"""A slowly moving copy of a module (exponential moving average of its weights), never trained by gradients."""

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class EmaCopy:
    """A frozen copy of ``online`` (eval mode, no gradients) that follows it slowly.

    ``update`` sets every weight to ``decay * copy + (1 - decay) * online``; buffers are
    copied as they are. ``decay = 0`` follows the online module exactly, ``decay = 1`` never moves.
    """

    def __init__(self, online: nn.Module, decay: float) -> None:
        self._online = online
        self._decay = decay
        self.module = copy.deepcopy(online)
        self.module.eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self) -> None:
        for average, current in zip(self.module.parameters(), self._online.parameters(), strict=True):
            average.lerp_(current, 1.0 - self._decay)
        for average, current in zip(self.module.buffers(), self._online.buffers(), strict=True):
            average.copy_(current)

    def reset(self) -> None:
        """Become an exact copy of the online module again."""
        self.module.load_state_dict(self._online.state_dict())

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        return self.module(x)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state)
