"""Restoring an optimizer from a checkpoint: the saved moments, the current run's settings.

A saved optimizer state numbers the parameters in the order its optimizer held them, which
differs between a run with weight decay (decayed weights first) and one without. A fork may
switch weight decay on, off or to another value, so each parameter's saved state (Adam's
step count and two moment estimates, the same for Adam and AdamW) is handed to the same
parameter of the new optimizer, whatever its position there. The hyperparameters (learning
rate, weight decay, epsilon) stay those the new optimizer was built with: they belong to
the run's config, not to the checkpoint.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from attentive_snake.training.optimizer import decays, parameter_order


def restore_optimizer(optimizer: torch.optim.Optimizer, saved: Mapping[str, Any], policy: nn.Module) -> None:
    """Load the per-parameter state of ``saved`` (an optimizer ``state_dict`` of the same ``policy`` architecture) into ``optimizer``."""
    saved_position = {id(parameter): index for index, parameter in enumerate(parameter_order(policy, decays(saved["param_groups"])))}
    own = optimizer.state_dict()
    own_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(own_parameters) != len(saved_position) or sum(len(group["params"]) for group in saved["param_groups"]) != len(saved_position):
        raise ValueError("the checkpoint's optimizer state does not fit this network (a different number of parameters).")
    state = {
        index: saved["state"][saved_position[id(parameter)]]
        for index, parameter in enumerate(own_parameters)
        if saved_position[id(parameter)] in saved["state"]
    }
    optimizer.load_state_dict({"state": state, "param_groups": own["param_groups"]})
