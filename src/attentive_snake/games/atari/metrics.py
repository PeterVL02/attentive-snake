"""Atari's own evaluation statistics, the same for every title: timeouts, lives lost, how often points came.

Pure functions of ``EpisodeRecord``s; events are those of ``outcome.py``. The last step of a
game carries its outcome instead of what happened in it, so that step counts in neither mean.
"""

from collections.abc import Sequence

from attentive_snake.episode_metrics import EpisodeRecord, GameMetrics, MetricDefinition, Statistics, mean, rate
from attentive_snake.games.atari.outcome import LIFE_LOST, SCORED, TIMEOUT

DEFINITIONS = (
    MetricDefinition("timeout_rate", "timeout rate", "Share of games cut off by the game's step limit instead of ending by the title's own rules.", None, "rate"),
    MetricDefinition("mean_lives_lost", "mean lives lost", "Average number of lives lost per game before its last step (always 0 in titles without lives, such as Pong).", False),
    MetricDefinition("mean_scoring_steps", "mean scoring steps", "Average number of steps per game on which the score went up (the last step not counted). In titles where points are rare (Montezuma's Revenge, Pitfall!) this shows whether the model finds any at all.", True),
)


def atari_statistics(records: Sequence[EpisodeRecord]) -> Statistics:
    def per_game(event: str) -> float | None:
        return mean([sum(step == event for step in record.events) for record in records])

    return {
        "timeout_rate": rate(sum(record.outcome == TIMEOUT for record in records), len(records)),
        "mean_lives_lost": per_game(LIFE_LOST),
        "mean_scoring_steps": per_game(SCORED),
    }


ATARI_METRICS = GameMetrics(definitions=DEFINITIONS, compute=atari_statistics)
