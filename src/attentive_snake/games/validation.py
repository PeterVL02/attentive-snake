"""Checks that a ``GameSpec`` is complete, so a broken game package fails loudly with its name."""

import dataclasses
import re

from attentive_snake.games.spec import GameSpec
from attentive_snake.observation_spec import OBSERVATION_KINDS
from attentive_snake.views import VIEW_KINDS

GAME_NAME = re.compile(r"[a-z0-9_]+")
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
    if spec.observation_labels is not None and not problems:
        problems += _label_problems(spec)
    return problems


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


def _dataclass_problems(cls: type, label: str) -> list[str]:
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        return [f"{label} must be a dataclass"]
    return [
        f"{label} field {f.name!r} needs a metadata['help'] description"
        for f in dataclasses.fields(cls)
        if not str(f.metadata.get("help", "")).strip()
    ]
