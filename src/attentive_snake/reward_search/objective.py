"""Which evaluation statistic judges the members of a reward search: one of the game's own, never one that measures the reward."""

from attentive_snake.episode_metrics import MetricDefinition
from attentive_snake.evaluation.statistics import metric_definitions
from attentive_snake.games.spec import GameSpec


def objective_choices(game: GameSpec) -> tuple[MetricDefinition, ...]:
    """The statistics a search of ``game`` may optimise: those with a direction that do not depend on the reward."""
    return tuple(d for d in metric_definitions(game) if d.higher_is_better is not None and not d.measures_reward)


def objective_definition(game: GameSpec, key: str | None) -> MetricDefinition:
    """The definition of ``key`` (``None``: the game's ``search_objective``); ``ValueError`` if it may not be optimised."""
    key = game.search_objective if key is None else key
    for definition in objective_choices(game):
        if definition.key == key:
            return definition
    offered = ", ".join(d.key for d in objective_choices(game))
    raise ValueError(f"objective {key!r} cannot judge a {game.name} reward search; choose one of {offered}.")
