"""The one place a learner's optimizer is built: Adam, or AdamW when ``train.weight_decay`` is above 0."""

import torch
from torch import nn

from attentive_snake.training.config import TrainConfig
from attentive_snake.training.weight_decay import decay_groups


def decays(optimizer_groups: list[dict]) -> bool:
    """Whether an optimizer (its ``param_groups``, live or saved) was built with weight decay: two groups instead of one."""
    return len(optimizer_groups) > 1


def build_optimizer(policy: nn.Module, train: TrainConfig, eps: float) -> torch.optim.Optimizer:
    """The optimizer of ``policy`` for ``train.learning_rate`` and ``train.weight_decay``; ``eps`` is the algorithm's Adam epsilon.

    Without weight decay it is plain Adam over ``policy.parameters()`` in one group, exactly
    what the learners built before the setting existed. With it, AdamW (the decay is applied
    to the weights directly, not mixed into the gradient that Adam rescales) over two groups:
    the decayed weights, then the exempt parameters with decay 0 (``training.weight_decay``).
    """
    if train.weight_decay == 0.0:
        return torch.optim.Adam(policy.parameters(), lr=train.learning_rate, eps=eps)
    groups = decay_groups(policy)
    return torch.optim.AdamW(
        [
            {"params": list(groups.decayed), "weight_decay": train.weight_decay},
            {"params": list(groups.exempt), "weight_decay": 0.0},
        ],
        lr=train.learning_rate,
        eps=eps,
    )


def parameter_order(policy: nn.Module, with_decay: bool) -> tuple[nn.Parameter, ...]:
    """The order ``build_optimizer`` numbers ``policy``'s parameters in, with or without weight decay."""
    return decay_groups(policy).ordered() if with_decay else tuple(policy.parameters())
