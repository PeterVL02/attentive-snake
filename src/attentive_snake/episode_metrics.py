"""Contract between games and evaluation: what one evaluated episode looks like, and how a game adds statistics.

Pure data, no numpy or torch, so game packages can import it cheaply.

* ``EpisodeRecord``: one finished evaluation episode, step by step (produced by
  ``attentive_snake.evaluation.player``).
* ``MetricDefinition``: how to label, format and judge one named statistic, so the app and
  the CLI can show any game's statistics without knowing the game.
* ``GameMetrics``: what a game may add to the generic statistics
  (``GameSpec.episode_metrics``): its definitions and a pure function computing them.

A statistic that cannot be computed (a mean over no games, steps to win when no game was
won) is ``None`` and shown as N/A, never a made-up 0.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

type MetricFormat = Literal["number", "integer", "rate"]
type Statistics = dict[str, float | None]

STEP_LIMIT = "step_limit"  # outcome of an evaluation episode cut off by the evaluator's own step cap


@dataclass(frozen=True)
class EpisodeRecord:
    """One evaluation episode.

    ``events[i]`` is the ``EnvStep.event`` of step ``i + 1`` (so ``len(events) == length``);
    the last one is the outcome, unless the evaluator's step cap cut the episode off, in
    which case ``outcome`` is ``STEP_LIMIT`` and the last event is whatever happened then.
    ``score`` is the game's headline number at the end, ``total_reward`` the summed reward.
    """

    score: float
    length: int
    total_reward: float
    outcome: str
    events: tuple[str, ...]


@dataclass(frozen=True)
class MetricDefinition:
    """How to present one statistic.

    Attributes:
        key: Name in the statistics mapping, ``[a-z0-9_]``.
        label: Short human name, e.g. ``"win rate"``.
        description: One plain sentence: what it measures and over which games.
        higher_is_better: ``True``/``False``, or ``None`` when neither is better (e.g. an
            episode length that is good for one game and bad for another).
        format: ``"rate"``: a fraction in [0, 1] shown as a percentage; ``"integer"``: a count;
            ``"number"``: anything else.
        unit: Shown after the value, e.g. ``"steps"`` (empty: none).
    """

    key: str
    label: str
    description: str
    higher_is_better: bool | None = True
    format: MetricFormat = "number"
    unit: str = ""


@dataclass(frozen=True)
class GameMetrics:
    """A game's own evaluation statistics.

    ``compute(records)`` is pure and returns exactly one value (or ``None``) per key of
    ``definitions``; it is called with at least one record.
    """

    definitions: tuple[MetricDefinition, ...]
    compute: Callable[[Sequence[EpisodeRecord]], Mapping[str, float | None]]


def rate(count: int, total: int) -> float | None:
    """``count / total``, ``None`` when there is nothing to divide by."""
    return count / total if total else None


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, ``None`` for no values."""
    return sum(values) / len(values) if values else None


def median(values: Sequence[float]) -> float | None:
    """Median (the mean of the middle two for an even count), ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
