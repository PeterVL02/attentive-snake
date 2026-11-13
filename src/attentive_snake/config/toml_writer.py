"""A tiny TOML serializer for documents made of flat tables of scalars.

Only what the experiment configs need: ``[section]`` headers followed by
``key = value`` lines, where a value is an int, float, bool or str. Nested
tables, arrays, dates and ``None`` (TOML has no null) are rejected.
"""

import json
import math
from collections.abc import Mapping

type Scalar = int | float | bool | str


def format_value(value: Scalar) -> str:
    """TOML literal of one scalar."""
    if isinstance(value, bool):  # before int: bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)  # "0.5", "1e-05", "2.0": all valid TOML floats
    if isinstance(value, str):
        # A JSON string (with non-ASCII kept) is a valid TOML basic string.
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"Cannot write {type(value).__name__} value {value!r} to TOML; expected int, float, bool or str.")


def format_tables(tables: Mapping[str, Mapping[str, Scalar]]) -> str:
    """Serialise ``{section: {key: scalar}}`` into a TOML document (sections in the given order)."""
    blocks = []
    for section, table in tables.items():
        lines = [f"[{section}]"]
        lines.extend(f"{key} = {format_value(value)}" for key, value in table.items())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
