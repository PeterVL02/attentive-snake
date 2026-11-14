"""Moving runs from the old flat layout (``runs/<name>/``) into a game directory (``runs/snake/<name>/``).

Before there were several games every run sat directly in the runs root; all of those are
Snake runs. ``plan_migration`` lists the moves without touching anything; ``migrate``
performs them. Nothing is overwritten: if any target exists, nothing is moved at all.
Running it again after a successful migration finds nothing to do.
"""

from pathlib import Path

from attentive_snake.run_dir import game_runs_dir, is_run_dir

LEGACY_GAME = "snake"

type Move = tuple[Path, Path]


def legacy_runs(root: Path) -> list[Path]:
    """Run directories (with a config.toml) directly in ``root``, by name."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((path for path in root.iterdir() if is_run_dir(path)), key=lambda path: path.name)


def plan_migration(root: Path, game: str = LEGACY_GAME) -> list[Move]:
    """``[(source, target)]`` for every legacy run; ``FileExistsError`` naming every target that already exists."""
    target_dir = game_runs_dir(root, game)
    moves = [(source, target_dir / source.name) for source in legacy_runs(root)]
    if any(source == target_dir for source, _ in moves):
        raise FileExistsError(f"{target_dir} is itself a run; rename it before migrating.")
    taken = [str(target) for _, target in moves if target.exists()]
    if taken:
        raise FileExistsError(f"refusing to overwrite: {', '.join(taken)} already exist(s); nothing was moved.")
    return moves


def migrate(root: Path, game: str = LEGACY_GAME) -> list[Move]:
    """Move every legacy run into ``root/<game>/``; returns the moves made (empty if there was nothing to do)."""
    moves = plan_migration(root, game)
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    return moves
