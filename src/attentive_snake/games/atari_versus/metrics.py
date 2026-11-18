"""Two-player Atari's own evaluation statistics: how often the model beat the random player, drew, or lost.

Pure functions of ``EpisodeRecord``s. A game counts by its score, the model's points minus the
opponent's when it ended, whether the title ended it, the step limit cut it (``timeout``) or the
evaluator's cap did (``step_limit``); so the three rates always add up to 1.
"""

from collections.abc import Sequence

from attentive_snake.episode_metrics import EpisodeRecord, GameMetrics, MetricDefinition, Statistics, rate
from attentive_snake.games.atari_versus.outcome import TIMEOUT

DEFINITIONS = (
    MetricDefinition("win_rate", "win rate", "Share of games against the random player that the model finished ahead on points (when the title ended the game, or when a step limit cut it off). The random player wins now and then, so a model that has learned nothing scores about as often as it concedes.", True, "rate"),
    MetricDefinition("draw_rate", "draw rate", "Share of games against the random player that ended level on points.", None, "rate"),
    MetricDefinition("loss_rate", "loss rate", "Share of games against the random player that the model ended behind on points.", False, "rate"),
    MetricDefinition("timeout_rate", "timeout rate", "Share of games cut off by the game's step limit (game.max_episode_steps) instead of ending by the title's own rules.", None, "rate"),
)


def versus_statistics(records: Sequence[EpisodeRecord]) -> Statistics:
    games = len(records)
    return {
        "win_rate": rate(sum(record.score > 0 for record in records), games),
        "draw_rate": rate(sum(record.score == 0 for record in records), games),
        "loss_rate": rate(sum(record.score < 0 for record in records), games),
        "timeout_rate": rate(sum(record.outcome == TIMEOUT for record in records), games),
    }


VERSUS_METRICS = GameMetrics(definitions=DEFINITIONS, compute=versus_statistics)
