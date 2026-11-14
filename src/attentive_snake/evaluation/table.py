"""An evaluation as plain text for the terminal: one line per statistic with its label, then the outcome counts."""

from collections.abc import Sequence

from attentive_snake.episode_metrics import MetricDefinition
from attentive_snake.evaluation.record import Evaluation

NOT_AVAILABLE = "N/A"


def format_value(definition: MetricDefinition, value: float | None) -> str:
    """``N/A`` for ``None``; rates as percentages, counts as whole numbers, the rest with two decimals; plus the unit."""
    if value is None:
        return NOT_AVAILABLE
    if definition.format == "rate":
        text = f"{100 * value:.1f} %"
    elif definition.format == "integer":
        text = f"{round(value):,}"
    else:
        text = f"{value:,.2f}"
    return f"{text} {definition.unit}" if definition.unit else text


def heading(evaluation: Evaluation) -> str:
    settings = evaluation.settings
    weights = evaluation.checkpoint or "the training weights"
    return (
        f"{weights} ({evaluation.env_steps:,} env steps): {settings.games} games, seed {settings.seed}, "
        f"greedy actions, step cap {settings.max_episode_steps:,}, on {settings.device}, {evaluation.seconds:.1f} s"
    )


def evaluation_table(evaluation: Evaluation, definitions: Sequence[MetricDefinition]) -> str:
    width = max(len(definition.label) for definition in definitions)
    lines = [heading(evaluation)]
    for definition in definitions:
        value = evaluation.statistics.get(definition.key)
        lines.append(f"  {definition.label:<{width}}  {format_value(definition, value)}")
    outcomes = ", ".join(f"{outcome} {count}" for outcome, count in evaluation.outcomes.items())
    lines.append(f"  {'outcomes':<{width}}  {outcomes}")
    return "\n".join(lines)
