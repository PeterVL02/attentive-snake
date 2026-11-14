"""A JSON-serialisable description of every setting of an experiment of one game, for UIs.

Field descriptions come from the config dataclasses' ``metadata["help"]``; the
per-kind facts from ``attentive_snake.models.registry.KIND_INFO``, told for the game's
observation kind (description, fields that matter, whether the kind is recommended). ``[game]`` and
``[reward]`` rows are the chosen game's own fields (``[game] name`` is not a row: it is
the top-level ``"game"`` of the schema).
"""

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any

from attentive_snake.config.experiment import SECTIONS, section_types
from attentive_snake.games.registry import DEFAULT_GAME, get_game
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS


def experiment_schema(game: str = DEFAULT_GAME) -> dict[str, Any]:
    """``{"game": name, "sections": {section: [field rows]}, "model_kinds": {kind: facts}}``; ``ValueError`` for an unknown game."""
    section_classes = section_types(game)
    observation_kind = get_game(game).observation_kind
    return {
        "game": game,
        "sections": {section: _field_rows(section_classes[section]) for section in SECTIONS},
        "model_kinds": {kind: _kind_row(kind, observation_kind) for kind in POLICY_KINDS},
    }


def _field_rows(cls: type) -> list[dict[str, Any]]:
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
        "choices": list(POLICY_KINDS) if (cls.__name__, field.name) == ("ModelConfig", "kind") else None,
    }


def _kind_row(kind: str, observation_kind: str) -> dict[str, Any]:
    """Facts about ``kind`` as it reads the game's ``observation_kind``."""
    info = KIND_INFO[kind]
    reading = info.inputs.get(observation_kind)
    return {
        "description": reading.description if reading else f"Cannot read {observation_kind} observations.",
        "observation_kinds": list(info.observation_kinds),
        "recommended": bool(reading and reading.recommended),
        "fields": list(info.fields_for(observation_kind)),
        "uses_history": info.uses_history,
        "recommended_env": {"history_length": 4 if info.uses_history else 1, "history_actions": info.uses_history},
        "attention_view": info.attention_view,
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
