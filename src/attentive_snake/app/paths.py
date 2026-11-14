"""Where the app finds and keeps its files. Injectable so tests can point it at temporary directories."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class AppPaths:
    """
    Attributes:
        runs_dir: Runs root; a game's runs are in ``runs_dir/<game>/<run>/``.
        configs_dir: Configs root; per game ``<game>/default.toml`` (pre-fills the
            new-run form) and ``<game>/rewards/*.toml`` (reward schemes), see ``config.layout``.
        train_script: ``scripts/train.py``, launched once per training job.
        cheat_sheet: Markdown file shown in the CLI tab.
        working_dir: Working directory of the training subprocesses.
    """

    runs_dir: Path
    configs_dir: Path
    train_script: Path
    cheat_sheet: Path
    working_dir: Path


def default_paths(runs_dir: Path | None = None, configs_dir: Path | None = None) -> AppPaths:
    """Paths inside this project checkout; ``runs_dir``/``configs_dir`` override the defaults."""
    return AppPaths(
        runs_dir=Path(runs_dir or PROJECT_ROOT / "runs").resolve(),
        configs_dir=Path(configs_dir or PROJECT_ROOT / "configs").resolve(),
        train_script=PROJECT_ROOT / "scripts" / "train.py",
        cheat_sheet=PROJECT_ROOT / "docs" / "CLI.md",
        working_dir=PROJECT_ROOT,
    )
