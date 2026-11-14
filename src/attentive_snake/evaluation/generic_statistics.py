"""The statistics every game's evaluation has: scores, lengths, returns, how many games were played."""

from collections import Counter
from collections.abc import Sequence

from attentive_snake.episode_metrics import EpisodeRecord, MetricDefinition, Statistics, mean, median

GENERIC_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("games", "games", "Number of games played in this evaluation.", None, "integer"),
    MetricDefinition("mean_score", "mean score", "Average of the game's score at the end of each game.", True),
    MetricDefinition("median_score", "median score", "Middle value of the end-of-game scores.", True),
    MetricDefinition("min_score", "lowest score", "Worst end-of-game score of any game.", True),
    MetricDefinition("max_score", "highest score", "Best end-of-game score of any game.", True),
    MetricDefinition("mean_length", "mean episode length", "Average number of steps per game.", None, unit="steps"),
    MetricDefinition("mean_return", "mean return", "Average summed training reward per game (the reward the model was trained on).", True),
)
GENERIC_KEYS = frozenset(definition.key for definition in GENERIC_METRICS)


def generic_statistics(records: Sequence[EpisodeRecord]) -> Statistics:
    """One value per ``GENERIC_METRICS`` key; means and extremes are ``None`` without records."""
    scores = [record.score for record in records]
    return {
        "games": float(len(records)),
        "mean_score": mean(scores),
        "median_score": median(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "mean_length": mean([record.length for record in records]),
        "mean_return": mean([record.total_reward for record in records]),
    }


def outcome_counts(records: Sequence[EpisodeRecord]) -> dict[str, int]:
    """``{outcome: number of games}``, most frequent first (ties alphabetical)."""
    counts = Counter(record.outcome for record in records)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
