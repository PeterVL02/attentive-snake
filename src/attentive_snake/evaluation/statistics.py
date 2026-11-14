"""All statistics of a game's evaluation: the generic ones plus the game's own (``GameSpec.episode_metrics``)."""

from collections.abc import Sequence

from attentive_snake.episode_metrics import EpisodeRecord, MetricDefinition, Statistics
from attentive_snake.evaluation.generic_statistics import GENERIC_METRICS, generic_statistics
from attentive_snake.games.spec import GameSpec


def metric_definitions(game: GameSpec) -> tuple[MetricDefinition, ...]:
    """The game's own definitions first (they are what it is about), then the generic ones."""
    own = game.episode_metrics.definitions if game.episode_metrics is not None else ()
    return (*own, *GENERIC_METRICS)


def evaluation_statistics(game: GameSpec, records: Sequence[EpisodeRecord]) -> Statistics:
    """``{key: value or None}`` for every definition of ``metric_definitions(game)``, in that order.

    ``ValueError`` if the game's function leaves out a key it defines or returns one it does not.
    """
    values = generic_statistics(records)
    if game.episode_metrics is not None:
        own = dict(game.episode_metrics.compute(records)) if records else {}
        defined = {definition.key for definition in game.episode_metrics.definitions}
        if records and set(own) != defined:
            raise ValueError(f"{game.name}'s episode_metrics returned {sorted(own)}, but defines {sorted(defined)}.")
        values |= {key: _plain(own.get(key)) for key in defined}
    return {definition.key: values[definition.key] for definition in metric_definitions(game)}


def _plain(value: float | None) -> float | None:
    """A plain float (not a numpy scalar), or ``None``."""
    return None if value is None else float(value)
