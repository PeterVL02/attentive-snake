"""Lunar Lander's own evaluation statistics: how each game ended, and how long a landing took.

Pure functions of ``EpisodeRecord``s; outcomes are those of ``outcome.py``. Rates are over
all games (a game cut off by the evaluator's step cap counts in none of them); steps to
land are over landed games only, N/A if none landed.
"""

from collections.abc import Sequence

from attentive_snake.episode_metrics import EpisodeRecord, GameMetrics, MetricDefinition, Statistics, mean, rate
from attentive_snake.games.lunar_lander.outcome import CRASHED, FLEW_OFF, LANDED, TIMEOUT

DEFINITIONS = (
    MetricDefinition("landing_rate", "landing rate", "Share of games that ended with the lander at rest (anywhere, not only between the flags).", True, "rate"),
    MetricDefinition("crash_rate", "crash rate", "Share of games that ended with the lander's body hitting the ground.", False, "rate"),
    MetricDefinition("flew_off_rate", "flew-off rate", "Share of games in which the lander left the screen sideways.", False, "rate"),
    MetricDefinition("timeout_rate", "timeout rate", "Share of games cut off by the game's time limit (usually hovering).", False, "rate"),
    MetricDefinition("mean_steps_to_land", "mean steps to land", "Average length of the games that ended landed (N/A when none landed).", False, unit="steps"),
)


def lander_statistics(records: Sequence[EpisodeRecord]) -> Statistics:
    def share(outcome: str) -> float | None:
        return rate(sum(record.outcome == outcome for record in records), len(records))

    return {
        "landing_rate": share(LANDED),
        "crash_rate": share(CRASHED),
        "flew_off_rate": share(FLEW_OFF),
        "timeout_rate": share(TIMEOUT),
        "mean_steps_to_land": mean([record.length for record in records if record.outcome == LANDED]),
    }


LANDER_METRICS = GameMetrics(definitions=DEFINITIONS, compute=lander_statistics)
