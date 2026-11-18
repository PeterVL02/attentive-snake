"""Checks that a ``GameSpec`` is complete, so a broken game package fails loudly with its name."""

import dataclasses
import re

from attentive_snake.episode_metrics import GameMetrics
from attentive_snake.evaluation.generic_statistics import GENERIC_KEYS, GENERIC_METRICS
from attentive_snake.games.spec import GameSpec
from attentive_snake.observation_spec import OBSERVATION_KINDS
from attentive_snake.views import VIEW_KINDS

GAME_NAME = re.compile(r"[a-z0-9_]+")
METRIC_KEY = re.compile(r"[a-z0-9_]+")
METRIC_FORMATS = ("number", "integer", "rate")
RESERVED_RULE_FIELDS = frozenset({"name"})  # [game] name selects the game


def spec_problems(spec: GameSpec, package_name: str) -> list[str]:
    """Everything wrong with ``spec`` exported by sub-package ``package_name`` (empty if nothing)."""
    problems = []
    if spec.name != package_name:
        problems.append(f"GAME.name {spec.name!r} must equal the package name {package_name!r}")
    if not GAME_NAME.fullmatch(spec.name):
        problems.append(f"GAME.name {spec.name!r} may only use a-z, 0-9 and '_'")
    if spec.observation_kind not in OBSERVATION_KINDS:
        problems.append(f"observation_kind {spec.observation_kind!r} is not one of {', '.join(OBSERVATION_KINDS)}")
    if spec.view_kind not in VIEW_KINDS:
        problems.append(f"view_kind {spec.view_kind!r} is not one of {', '.join(VIEW_KINDS)}")
    if not callable(spec.make_env):
        problems.append("make_env must be callable")
    problems += _dataclass_problems(spec.rules_type, "rules_type")
    problems += _dataclass_problems(spec.reward_type, "reward_type")
    if dataclasses.is_dataclass(spec.rules_type):
        rule_fields = {f.name for f in dataclasses.fields(spec.rules_type)}
        problems += [f"rules_type may not have a field {name!r} (reserved)" for name in RESERVED_RULE_FIELDS & rule_fields]
        problems += [f"locked_rules names {name!r}, which is not a rules field" for name in spec.locked_rules if name not in rule_fields]
    if spec.episode_metrics is not None:
        problems += metric_problems(spec.episode_metrics)
    problems += _search_objective_problems(spec)
    if spec.training_warnings is not None and not callable(spec.training_warnings):
        problems.append("training_warnings must be callable")
    if spec.continuous_actions is not None and not callable(spec.continuous_actions):
        problems.append("continuous_actions must be callable")
    problems += _seat_problems(spec)
    if spec.observation_labels is not None and not problems:
        problems += _label_problems(spec)
    if spec.action_labels is not None and not problems:
        problems += _action_label_problems(spec)
    return problems


def metric_problems(metrics: GameMetrics) -> list[str]:
    """``episode_metrics``: callable, distinct well-formed keys that do not shadow a generic statistic, labels and descriptions."""
    if not isinstance(metrics, GameMetrics):
        return ["episode_metrics must be a GameMetrics"]
    problems = [] if callable(metrics.compute) else ["episode_metrics.compute must be callable"]
    keys = [definition.key for definition in metrics.definitions]
    if len(set(keys)) != len(keys):
        problems.append("episode_metrics keys must be distinct")
    for definition in metrics.definitions:
        if not METRIC_KEY.fullmatch(definition.key):
            problems.append(f"episode_metrics key {definition.key!r} may only use a-z, 0-9 and '_'")
        if definition.key in GENERIC_KEYS:
            problems.append(f"episode_metrics key {definition.key!r} is a generic statistic every game has")
        if not (definition.label.strip() and definition.description.strip()):
            problems.append(f"episode_metrics {definition.key!r} needs a label and a description")
        if definition.format not in METRIC_FORMATS:
            problems.append(f"episode_metrics {definition.key!r}: format must be one of {', '.join(METRIC_FORMATS)}")
    return problems


def _search_objective_problems(spec: GameSpec) -> list[str]:
    """``search_objective``: one of the game's statistics, with a direction, not measuring the reward."""
    own = spec.episode_metrics.definitions if isinstance(spec.episode_metrics, GameMetrics) else ()
    definition = next((d for d in (*own, *GENERIC_METRICS) if d.key == spec.search_objective), None)
    if definition is None:
        return [f"search_objective {spec.search_objective!r} is not one of the game's evaluation statistics"]
    if definition.higher_is_better is None or definition.measures_reward:
        return [f"search_objective {spec.search_objective!r} must have a direction and must not measure the reward"]
    return []


def _seat_problems(spec: GameSpec) -> list[str]:
    """``seats`` and ``make_match``: a positive count, and a callable ``make_match`` exactly when there are several seats."""
    if not isinstance(spec.seats, int) or spec.seats < 1:
        return ["seats must be a positive whole number"]
    if spec.seats > 1 and not callable(spec.make_match):
        return [f"a game with {spec.seats} seats needs a callable make_match (its self-play match)"]
    if spec.seats == 1 and spec.make_match is not None:
        return ["make_match is only for games with several seats (seats > 1)"]
    return []


def _label_problems(spec: GameSpec) -> list[str]:
    """``observation_labels``: distinct non-empty names, one per number of a default env's frame (vector games only)."""
    labels = spec.observation_labels
    if spec.observation_kind != "vector":
        return [f"observation_labels are only for vector games, not {spec.observation_kind} ones"]
    if not all(isinstance(label, str) and label.strip() for label in labels) or len(set(labels)) != len(labels):
        return ["observation_labels must be distinct, non-empty strings"]
    try:
        shape = tuple(spec.make_env(spec.rules_type(), spec.reward_type(), 0).observation_shape)
    except Exception as error:
        return [f"cannot build a default env to check observation_labels: {error!r}"]
    if shape != (len(labels),):
        return [f"observation_labels has {len(labels)} names but a frame has shape {shape}"]
    return []


def _action_label_problems(spec: GameSpec) -> list[str]:
    """``action_labels``: distinct non-empty names, one per action of a default env."""
    labels = spec.action_labels
    if not all(isinstance(label, str) and label.strip() for label in labels) or len(set(labels)) != len(labels):
        return ["action_labels must be distinct, non-empty strings"]
    try:
        num_actions = spec.make_env(spec.rules_type(), spec.reward_type(), 0).num_actions
    except Exception as error:
        return [f"cannot build a default env to check action_labels: {error!r}"]
    if num_actions != len(labels):
        return [f"action_labels has {len(labels)} names but the game has {num_actions} actions"]
    return []


def _dataclass_problems(cls: type, label: str) -> list[str]:
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        return [f"{label} must be a dataclass"]
    return [
        f"{label} field {f.name!r} needs a metadata['help'] description"
        for f in dataclasses.fields(cls)
        if not str(f.metadata.get("help", "")).strip()
    ]
