"""Process-level setup: picking a torch device and seeding random number generators."""

import random

import numpy as np
import torch


def resolve_device(name: str) -> torch.device:
    """Map "auto" to cuda when available (else cpu); pass anything else to torch."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
