"""Command lines of ``scripts/train.py`` for a new run and for a resumed run."""

import sys
from pathlib import Path


def _python(script: Path) -> list[str]:
    # -u: unbuffered, so train.log fills while the run is going.
    return [sys.executable, "-u", str(script)]


def new_run_command(script: Path, run_dir: Path, config_file: Path) -> list[str]:
    return [*_python(script), "--run-dir", str(run_dir), "--config", str(config_file)]


def resume_command(script: Path, run_dir: Path, total_steps: int | None = None) -> list[str]:
    steps = [] if total_steps is None else ["--steps", str(total_steps)]
    return [*_python(script), "--resume", str(run_dir), *steps]
