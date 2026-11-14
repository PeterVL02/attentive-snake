"""``TrainingMetric``: one number a run logs in ``metrics.jsonl`` while training, with how to show it.

Torch-free. Algorithms declare theirs (``training.algorithm.Algorithm.metrics``), model kinds
with an auxiliary objective theirs (``models.kind_info.KindInfo.metrics``); the schema passes
both to the app, which charts a metric only for runs that log it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingMetric:
    """``key`` in ``metrics.jsonl``, a short ``label`` and a plain ``description``.

    ``setting`` is ``"section.field"`` of the run's config that switches the metric on: it is
    logged (and charted) only by runs where that field is on (true, or a number above 0);
    ``None`` = always logged by the runs the metric belongs to.
    """

    key: str
    label: str
    description: str
    setting: str | None = None


def setting_is_on(value: object) -> bool:
    """Whether a config value switches a metric on: ``True`` or a number above 0."""
    if isinstance(value, bool):
        return value
    return isinstance(value, int | float) and value > 0
