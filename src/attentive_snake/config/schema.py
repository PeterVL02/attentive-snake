"""A JSON-serialisable description of every setting of an experiment of one game, for UIs.

Field descriptions come from the config dataclasses' ``metadata["help"]``; the
per-kind facts from ``attentive_snake.models.registry.KIND_INFO``, told for the game's
observation kind (description, fields that matter, whether the kind is recommended, the
training metrics its auxiliary objective logs); a kind that cannot read that observation
kind is not listed (``resnet`` and ``mlp`` for image games). ``[game]`` and
``[reward]`` rows are the chosen game's own fields (``[game] name`` is not a row: it is
the top-level ``"game"`` of the schema). Every algorithm's section is listed (``"ppo"``,
``"dqn"``: a UI shows the one ``train.algorithm`` selects), and ``"algorithms"`` tells, per
algorithm, its label, description, section, policy head (with its default settings; a
setting may choose another, e.g. ``dqn.dueling``) and the training metrics it logs.
"""

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any

from attentive_snake.config.experiment import SHARED_SECTIONS, section_types
from attentive_snake.games.registry import DEFAULT_GAME, get_game
from attentive_snake.models.config import VALUE_TRAINING
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS
from attentive_snake.training.algorithm import Algorithm
from attentive_snake.training.algorithms import algorithm_names, get_algorithm

_CHOICES = {("ModelConfig", "kind"): POLICY_KINDS, ("ModelConfig", "value_training"): VALUE_TRAINING, ("TrainConfig", "algorithm"): algorithm_names()}


def experiment_schema(game: str = DEFAULT_GAME) -> dict[str, Any]:
    """``{"game", "sections": {section: [field rows]}, "model_kinds": {kind: facts}, "algorithms": {name: facts}}``.

    ``ValueError`` for an unknown game.
    """
    section_classes = section_types(game)
    observation_kind = get_game(game).observation_kind
    algorithms = {name: get_algorithm(name) for name in algorithm_names()}
    return {
        "game": game,
        "sections": {
            **{section: field_rows(section_classes[section]) for section in SHARED_SECTIONS},
            **{name: field_rows(algorithm.settings_type) for name, algorithm in algorithms.items()},
        },
        "model_kinds": {kind: _kind_row(kind, observation_kind) for kind in POLICY_KINDS if observation_kind in KIND_INFO[kind].inputs},
        "algorithms": {name: _algorithm_row(algorithm) for name, algorithm in algorithms.items()},
    }


def _algorithm_row(algorithm: Algorithm) -> dict[str, Any]:
    return {
        "label": algorithm.label,
        "description": algorithm.description,
        "section": algorithm.name,
        "head": algorithm.head(algorithm.settings_type()),
        "metrics": [dataclasses.asdict(metric) for metric in algorithm.metrics],
    }


def field_rows(cls: type) -> list[dict[str, Any]]:
    """One form row per field of the flat dataclass ``cls`` (``name``, ``type``, ``default``, ``nullable``, ``description``, ``choices``)."""
    hints = typing.get_type_hints(cls)
    return [_field_row(field, hints[field.name], cls) for field in dataclasses.fields(cls)]


def _field_row(field: dataclasses.Field, annotation: Any, cls: type) -> dict[str, Any]:
    default = field.default
    return {
        "name": field.name,
        "type": _type_name(annotation),
        "default": default.as_posix() if isinstance(default, Path) else default,
        "nullable": _is_nullable(annotation),
        "description": field.metadata["help"],
        "choices": _choices(field, cls),
    }


def _choices(field: dataclasses.Field, cls: type) -> list[str] | None:
    """The allowed values: the table above, or a field's own ``metadata["choices"]`` (a game's rules)."""
    options = _CHOICES.get((cls.__name__, field.name), field.metadata.get("choices"))
    return None if options is None else list(options)


def _kind_row(kind: str, observation_kind: str) -> dict[str, Any]:
    """Facts about ``kind`` as it reads the game's ``observation_kind``."""
    info = KIND_INFO[kind]
    reading = info.inputs[observation_kind]
    return {
        "description": reading.description,
        "observation_kinds": list(info.observation_kinds),
        "recommended": reading.recommended,
        "fields": list(info.fields_for(observation_kind)),
        "uses_history": info.uses_history,
        "recommended_env": {"history_length": 4 if info.uses_history else 1, "history_actions": info.uses_history},
        "attention_view": info.attention_view,
        "metrics": [dataclasses.asdict(metric) for metric in info.metrics],
    }


def _members(annotation: Any) -> tuple[type, ...]:
    if isinstance(annotation, types.UnionType):
        return typing.get_args(annotation)
    return (annotation,)


def _is_nullable(annotation: Any) -> bool:
    return types.NoneType in _members(annotation)


def _type_name(annotation: Any) -> str:
    (base,) = [t for t in _members(annotation) if t is not types.NoneType]
    return "path" if base is Path else base.__name__
