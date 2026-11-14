"""Move runs from the old flat layout runs/<name>/ into runs/snake/<name>/ (all old runs are Snake runs).

    uv run python scripts/migrate_runs.py [--runs-dir runs] [--dry-run]

Only directories that directly contain a config.toml are moved. Nothing is overwritten:
if a target already exists, nothing is moved at all. Safe to run again.
"""

import argparse
from pathlib import Path

from attentive_snake.run_migration import LEGACY_GAME, migrate, plan_migration

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help=f"runs root (default: {DEFAULT_RUNS_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be moved")
    args = parser.parse_args()
    try:
        moves = plan_migration(args.runs_dir) if args.dry_run else migrate(args.runs_dir)
    except FileExistsError as error:
        parser.exit(1, f"error: {error}\n")
    verb = "would move" if args.dry_run else "moved"
    for source, target in moves:
        print(f"{verb} {source} -> {target}")
    done = "would be moved" if args.dry_run else "moved"
    print(f"{len(moves)} run(s) {done} into {args.runs_dir / LEGACY_GAME}" if moves else "nothing to migrate")


if __name__ == "__main__":
    main()
