"""Saving and restoring training state (model, optimizer, environment-step counter, the learner's extra state)."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


_CHECKPOINT_NAME = re.compile(r"checkpoint_(\d+)\.pt")


def checkpoint_path(directory: Path, env_steps: int) -> Path:
    return Path(directory) / f"checkpoint_{env_steps:010d}.pt"


def checkpoint_steps(path: Path) -> int:
    """Env steps encoded in a checkpoint file name (``ValueError`` if it is not one)."""
    match = _CHECKPOINT_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"{path} is not a checkpoint file name (checkpoint_<env_steps>.pt)")
    return int(match.group(1))


def list_checkpoints(directory: Path) -> list[Path]:
    """Checkpoint files in ``directory`` sorted by env steps, ascending (empty if none or no directory)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if _CHECKPOINT_NAME.fullmatch(path.name)), key=checkpoint_steps
    )


def latest_checkpoint(directory: Path) -> Path | None:
    """The checkpoint with the most env steps in ``directory`` (by file name), or ``None`` if there is none."""
    checkpoints = list_checkpoints(directory)
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(directory: Path, keep: Path) -> list[Path]:
    """Delete every checkpoint file in ``directory`` except ``keep``; returns the deleted paths.

    Only files named like checkpoints (``checkpoint_<env_steps>.pt``) are touched.
    """
    keep = Path(keep).resolve()
    deleted = [path for path in list_checkpoints(directory) if path.resolve() != keep]
    for path in deleted:
        path.unlink(missing_ok=True)
    return deleted


def save_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    env_steps: int,
    learner_state: Mapping[str, Any] | None = None,
) -> None:
    """``learner_state``: what the algorithm keeps besides policy and optimizer (``Learner.extra_state``), saved under ``"learner"``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_steps": env_steps,
        "learner": dict(learner_state or {}),
    }
    torch.save(state, path)


def read_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """The whole saved state: ``{"policy", "optimizer", "env_steps", "learner"}`` (tensors only, no code is loaded)."""
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def load_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Load state into ``policy`` (and ``optimizer`` if given); returns the saved env-step count."""
    state = read_checkpoint(path, map_location)
    policy.load_state_dict(state["policy"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["env_steps"])
