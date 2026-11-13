"""Which config fields differ between several experiment configs (pure function, JSON in and out)."""

from collections.abc import Mapping
from typing import Any

Tables = Mapping[str, Mapping[str, Any]]


def differing_fields(configs: Mapping[str, Tables]) -> list[dict[str, Any]]:
    """Fields whose value is not the same in every config.

    ``configs`` maps a name (e.g. a run) to its ``{section: {field: value}}`` tables. A field
    missing from a config counts as ``None``. Returns ``[{"section", "field", "values": {name: value}}]``
    in section and field order of first appearance; empty for fewer than two configs.
    """
    if len(configs) < 2:
        return []
    differences = []
    for section, field in _keys(configs.values()):
        values = {name: tables.get(section, {}).get(field) for name, tables in configs.items()}
        if len({_hashable(value) for value in values.values()}) > 1:
            differences.append({"section": section, "field": field, "values": values})
    return differences


def _keys(all_tables: Any) -> list[tuple[str, str]]:
    keys: dict[tuple[str, str], None] = {}
    for tables in all_tables:
        for section, table in tables.items():
            for field in table:
                keys[(section, field)] = None
    return list(keys)


def _hashable(value: Any) -> Any:
    # 1 and 1.0 (and True) compare equal in Python; a TOML int vs float is not a real difference,
    # but a bool vs a number is.
    return (type(value) is bool, value)
