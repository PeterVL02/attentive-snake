"""Contract between a reward search and whatever trains its member runs (the app: the training queue).

A search only needs to fork runs, resume one that ended early, cancel what it started, ask how far
a run has got and read its last evaluation made while training. Every answer is read from what
exists now (the queue, the jobs, the run directory), never from what an earlier process remembered,
so a search carries on after a restart of the app.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

# ``missing``: no run and nothing waiting for it; ``waiting``: in the training queue;
# ``training``: its job runs; ``ended``: its directory exists and no job runs (finished, failed, stopped, or the app restarted).
RunProgress = Literal["missing", "waiting", "training", "ended"]


class LaunchRefused(RuntimeError):
    """The fork or resume could not be put in the queue; the message says why."""


@dataclass(frozen=True)
class FinalEvaluation:
    """A run's last evaluation made while it trained (``evaluation.record.Evaluation``, the parts a search reads)."""

    env_steps: int
    checkpoint: str | None
    statistics: Mapping[str, float | None]


class TrainingBackend(Protocol):
    def run_tables(self, game: str, run: str) -> dict[str, dict[str, Any]]:
        """The run's experiment tables (its config.toml); ``LookupError`` if there is no such run."""
        ...

    def enqueue_fork(
        self, game: str, parent: str, checkpoint: str | None, run: str, tables: dict[str, dict[str, Any]], description: str
    ) -> None:
        """Queue a new run ``run`` forked from ``parent``'s ``checkpoint`` (``None``: its latest); ``LaunchRefused`` if refused."""
        ...

    def enqueue_resume(self, game: str, run: str) -> None:
        """Queue a resume of ``run`` to its own target; ``LaunchRefused`` if refused."""
        ...

    def progress(self, game: str, run: str) -> RunProgress: ...

    def last_training_evaluation(self, game: str, run: str) -> FinalEvaluation | None:
        """The run's newest evaluation made while training, ``None`` if it has none."""
        ...

    def cancel(self, game: str, run: str) -> None:
        """Take ``run``'s entries out of the queue and stop its job if one runs (nothing if neither)."""
        ...
