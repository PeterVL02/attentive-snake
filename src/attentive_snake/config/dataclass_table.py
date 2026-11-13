"""Converting flat dataclasses to and from TOML-style tables (``dict[str, scalar]``).

Supported field types: ``int``, ``float``, ``bool``, ``str``, ``pathlib.Path``
and any of them ``| None``.

* Writing: ``None`` values are omitted (TOML has no null); ``Path`` becomes a
  POSIX string. Omitting is only lossless when the field's default is ``None``
  too, so a ``None`` in a field with a non-``None`` default is an error.
* Reading: absent keys keep the dataclass defaults; unknown keys raise a
  ``ValueError`` naming the key and the valid keys; ints are accepted for float
  fields and strings for ``Path`` fields; any other type mismatch raises
  ``TypeError``. The dataclass's own ``__post_init__`` validation still runs.
"""

import dataclasses
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attentive_snake.config.toml_writer import Scalar

_SCALARS = (int, float, bool, str, Path, types.NoneType)


def dataclass_to_table(obj: Any) -> dict[str, Scalar]:
    """Field values of a dataclass instance as a flat table, in field order."""
    table: dict[str, Scalar] = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if value is None:
            if field.default is not None:
                raise ValueError(
                    f"{type(obj).__name__}.{field.name} is None but defaults to {field.default!r}; "
                    "None cannot be written to TOML, so it would not survive a round trip."
                )
            continue
        table[field.name] = value.as_posix() if isinstance(value, Path) else value
    return table


def dataclass_from_table[T](cls: type[T], table: Mapping[str, Any], section: str = "") -> T:
    """Build ``cls`` from ``table``; ``section`` only prefixes error messages."""
    where = f"[{section}] " if section else ""
    hints = typing.get_type_hints(cls)
    valid = [field.name for field in dataclasses.fields(cls)]
    kwargs = {}
    for key, value in table.items():
        if key not in valid:
            raise ValueError(f"{where}unknown key {key!r} for {cls.__name__}; valid keys: {', '.join(valid)}.")
        kwargs[key] = _coerce(value, _allowed_types(hints[key]), f"{where}{key}")
    return cls(**kwargs)


def _allowed_types(annotation: Any) -> tuple[type, ...]:
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        allowed = typing.get_args(annotation)
    else:
        allowed = (annotation,)
    unsupported = [t for t in allowed if t not in _SCALARS]
    if unsupported:
        raise TypeError(f"Unsupported field type {annotation!r} for a TOML table.")
    return allowed


def _coerce(value: Any, allowed: tuple[type, ...], name: str) -> Any:
    if isinstance(value, bool):
        if bool in allowed:
            return value
    elif isinstance(value, int) and int not in allowed and float in allowed:
        return float(value)
    elif isinstance(value, str) and Path in allowed:
        return Path(value)
    elif isinstance(value, tuple(t for t in allowed if t is not bool)):
        return value
    expected = " or ".join(t.__name__ for t in allowed if t is not types.NoneType)
    raise TypeError(f"{name} must be {expected}, got {type(value).__name__} {value!r}.")
