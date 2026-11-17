"""Which ``[reward]`` fields of a game a search can change, and checking a proposed table against the game's rules."""

import dataclasses
import typing
from collections.abc import Mapping
from typing import Any

from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.games.spec import GameSpec


def searchable_fields(game: GameSpec) -> tuple[str, ...]:
    """The reward coefficients (``float`` fields) of ``game``, in the table's order; switches and counts are not searched."""
    hints = typing.get_type_hints(game.reward_type)
    return tuple(field.name for field in dataclasses.fields(game.reward_type) if hints[field.name] is float)


def require_searchable(game: GameSpec, names: tuple[str, ...]) -> None:
    """``ValueError`` naming each of ``names`` that is not a searchable field of ``game``."""
    allowed = searchable_fields(game)
    unknown = [name for name in names if name not in allowed]
    if unknown:
        offered = ", ".join(allowed) or "none"
        raise ValueError(f"{game.name} has no reward coefficient(s) {', '.join(map(repr, unknown))} to search (it has: {offered}).")


def valid_reward(game: GameSpec, table: Mapping[str, Any]) -> bool:
    """Whether ``table`` is a reward the game accepts (its ``[reward]`` validation passes)."""
    try:
        dataclass_from_table(game.reward_type, table, section="reward")
    except (ValueError, TypeError):
        return False
    return True


def full_reward(game: GameSpec, table: Mapping[str, Any]) -> dict[str, Any]:
    """``table`` with every field of the game's reward filled in (defaults for the missing ones); ``ValueError`` if invalid."""
    return dataclass_to_table(dataclass_from_table(game.reward_type, table, section="reward"))
