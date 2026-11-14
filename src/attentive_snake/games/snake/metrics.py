"""Snake's own evaluation statistics: how often it wins, how fast it wins, and how fast it finds each apple.

Pure functions of ``EpisodeRecord``s (events are ``StepEvent`` values). An apple is eaten
on every ``"ate_food"`` step and on the ``"won"`` step (the last apple fills the board).

* Steps to win: the length of each won game; over won games only, N/A if none was won.
* Steps between apples: for every apple, the steps since the previous apple (or since the
  start, for a game's first apple). The steps after a game's last apple until it ended
  (death, starvation, the evaluator's step cap) are not an interval: no apple was reached.
  Pooled over all games; N/A if no apple was eaten at all.
"""

from collections.abc import Sequence

from attentive_snake.episode_metrics import EpisodeRecord, GameMetrics, MetricDefinition, Statistics, mean, median, rate
from attentive_snake.games.snake.types import StepEvent

APPLE_EVENTS = frozenset({StepEvent.ATE_FOOD.value, StepEvent.WON.value})

DEFINITIONS = (
    MetricDefinition("win_rate", "win rate", "Share of games in which the snake filled the whole board.", True, "rate"),
    MetricDefinition("mean_steps_to_win", "mean steps to win", "Average length of the games that were won (N/A when none was won).", False, unit="steps"),
    MetricDefinition("median_steps_to_win", "median steps to win", "Middle length of the games that were won (N/A when none was won).", False, unit="steps"),
    MetricDefinition(
        "mean_steps_between_apples",
        "mean steps between apples",
        "Average steps from one apple (or the start) to the next, over every apple of every game; "
        "the steps after a game's last apple do not count.",
        False,
        unit="steps",
    ),
    MetricDefinition(
        "median_steps_between_apples",
        "median steps between apples",
        "Middle value of the steps from one apple (or the start) to the next, over every apple of every game.",
        False,
        unit="steps",
    ),
)


def apple_steps(record: EpisodeRecord) -> list[int]:
    """The step numbers (1-based) at which apples were eaten."""
    return [step for step, event in enumerate(record.events, start=1) if event in APPLE_EVENTS]


def apple_intervals(record: EpisodeRecord) -> list[int]:
    """Steps from the start or the previous apple to each apple, in order."""
    steps = apple_steps(record)
    return [current - previous for previous, current in zip([0, *steps], steps)]


def snake_statistics(records: Sequence[EpisodeRecord]) -> Statistics:
    won = [record.length for record in records if record.outcome == StepEvent.WON.value]
    intervals = [interval for record in records for interval in apple_intervals(record)]
    return {
        "win_rate": rate(len(won), len(records)),
        "mean_steps_to_win": mean(won),
        "median_steps_to_win": median(won),
        "mean_steps_between_apples": mean(intervals),
        "median_steps_between_apples": median(intervals),
    }


SNAKE_METRICS = GameMetrics(definitions=DEFINITIONS, compute=snake_statistics)
