"""The target network: a frozen copy of the online network that supplies bootstrap values, refreshed by hard copies."""

import copy
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn


class TargetNetwork:
    """A copy of ``online`` in eval mode without gradients; ``sync`` copies the online weights into it."""

    def __init__(self, online: nn.Module) -> None:
        self.network = copy.deepcopy(online)
        self.network.eval()
        self.network.requires_grad_(False)

    def sync(self, online: nn.Module) -> None:
        self.network.load_state_dict(online.state_dict())

    def __call__(self, observations: Tensor) -> Any:
        return self.network(observations)

    def state_dict(self) -> dict[str, Any]:
        return self.network.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.network.load_state_dict(state)
