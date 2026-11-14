"""Single-setting overrides of an ``ExperimentConfig`` written as ``section.key=value`` (e.g. on a command line).

The value is read as a TOML value (``1e-4``, ``true``, ``"cpu"``); anything that is
not valid TOML is taken as a plain string, so ``train.device=cpu`` also works.
TOML has no null, so a setting cannot be overridden to ``None`` this way.
"""

import tomllib
from collections.abc import Iterable
from typing import Any

from attentive_snake.config.experiment import ExperimentConfig
from attentive_snake.config.experiment_io import experiment_from_tables, experiment_to_tables

type Override = tuple[str, str, Any]


def parse_override(text: str) -> Override:
    """``"train.learning_rate=1e-4"`` -> ``("train", "learning_rate", 0.0001)``; ``ValueError`` if malformed."""
    target, separator, raw = text.partition("=")
    section, dot, key = target.strip().partition(".")
    if not separator or not dot or not section or not key or "." in key:
        raise ValueError(f"invalid override {text!r}: write section.key=value, e.g. train.learning_rate=1e-4.")
    return section, key, _value(raw.strip())


def apply_overrides(config: ExperimentConfig, overrides: Iterable[Override]) -> ExperimentConfig:
    """``config`` with each ``(section, key, value)`` set; unknown sections/keys and bad values raise as in TOML loading."""
    tables = experiment_to_tables(config)
    for section, key, value in overrides:
        tables.setdefault(section, {})[key] = value
    return experiment_from_tables(tables)


def _value(raw: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw
