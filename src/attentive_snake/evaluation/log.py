"""A run's evaluations.jsonl: appending one evaluation per line, and reading them back."""

import json
from pathlib import Path

from attentive_snake.evaluation.record import Evaluation


def append_evaluation(path: Path, evaluation: Evaluation) -> None:
    """Append ``evaluation`` as one JSON line (``None`` as ``null``), flushed at once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(evaluation.to_json()) + "\n")


def read_evaluations(path: Path) -> list[Evaluation]:
    """Every evaluation in the file, in file order; ``[]`` if it is missing. Unreadable lines are skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    evaluations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            evaluations.append(Evaluation.from_json(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            continue
    return evaluations
