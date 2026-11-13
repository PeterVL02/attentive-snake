"""A JSON-serialisable description of every experiment setting, for UIs.

Field descriptions come from the config dataclasses' ``metadata["help"]``; the
per-kind facts from ``attentive_snake.models.registry.KIND_INFO``.
"""

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any

from attentive_snake.config.experiment import ExperimentConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS


def experiment_schema() -> dict[str, Any]:
    return {
        "sections": {
            section.name: _field_rows(section.type) for section in dataclasses.fields(ExperimentConfig)
        },
        "model_kinds": {kind: _kind_row(kind) for kind in POLICY_KINDS},
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


def _kind_row(kind: str) -> dict[str, Any]:
    info = KIND_INFO[kind]
    return {
        "description": info.description,
        "fields": list(info.fields),
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
