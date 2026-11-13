"""Where the app finds and keeps its files. Injectable so tests can point it at temporary directories."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class AppPaths:
    """
    Attributes:
        runs_dir: Parent directory of the run directories.
        rewards_dir: Directory of the saved reward schemes (``*.toml``).
        default_config: Experiment TOML whose values pre-fill the new-run form.
        train_script: ``scripts/train.py``, launched once per training job.
        cheat_sheet: Markdown file shown in the CLI tab.
        working_dir: Working directory of the training subprocesses.
    """

    runs_dir: Path
    rewards_dir: Path
    default_config: Path
    train_script: Path
    cheat_sheet: Path
    working_dir: Path


def default_paths(runs_dir: Path | None = None, rewards_dir: Path | None = None) -> AppPaths:
    """Paths inside this project checkout; ``runs_dir``/``rewards_dir`` override the defaults."""
    return AppPaths(
        runs_dir=Path(runs_dir or PROJECT_ROOT / "runs").resolve(),
        rewards_dir=Path(rewards_dir or PROJECT_ROOT / "configs" / "rewards").resolve(),
        default_config=PROJECT_ROOT / "configs" / "default.toml",
        train_script=PROJECT_ROOT / "scripts" / "train.py",
        cheat_sheet=PROJECT_ROOT / "docs" / "CLI.md",
        working_dir=PROJECT_ROOT,
    )
