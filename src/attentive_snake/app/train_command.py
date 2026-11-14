"""Command lines of ``scripts/train.py`` for a new run, a resumed run and a forked run."""

import sys
from pathlib import Path


def _python(script: Path) -> list[str]:
    # -u: unbuffered, so train.log fills while the run is going.
    return [sys.executable, "-u", str(script)]


def _description(description: str | None) -> list[str]:
    # One "--opt=value" argument, so a description starting with "-" is not read as an option.
    return [] if description is None else [f"--description={description}"]


def new_run_command(script: Path, run_dir: Path, config_file: Path, description: str | None = None) -> list[str]:
    return [*_python(script), "--run-dir", str(run_dir), "--config", str(config_file), *_description(description)]


def resume_command(
    script: Path,
    run_dir: Path,
    total_steps: int | None = None,
    keep_checkpoints: bool | None = None,
    description: str | None = None,
) -> list[str]:
    steps = [] if total_steps is None else ["--steps", str(total_steps)]
    keep = [] if keep_checkpoints is None else ["--keep-checkpoints" if keep_checkpoints else "--no-keep-checkpoints"]
    return [*_python(script), "--resume", str(run_dir), *steps, *keep, *_description(description)]


def fork_command(
    script: Path, run_dir: Path, parent_dir: Path, checkpoint: str, config_file: Path, description: str | None = None
) -> list[str]:
    return [
        *_python(script),
        "--run-dir", str(run_dir),
        "--fork", str(parent_dir),
        "--fork-checkpoint", checkpoint,
        "--config", str(config_file),
        *_description(description),
    ]
