"""``SearchSpec``: what one reward search is (its name, game, parent, fields and settings), as JSON-able data."""

from dataclasses import dataclass
from typing import Any

from attentive_snake.reward_search.settings import SearchSettings, settings_from_json, settings_to_json


@dataclass(frozen=True)
class SearchSpec:
    """
    Attributes:
        name: The search's name; its member runs are named after it (``naming``).
        game: The game of the parent and of every member run.
        parent: The run the search starts from.
        checkpoint: The parent's checkpoint file the first generation forks.
        fields: The ``[reward]`` fields the search may change, in the reward table's order.
        settings: The numbers it runs with.
    """

    name: str
    game: str
    parent: str
    checkpoint: str
    fields: tuple[str, ...]
    settings: SearchSettings

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("select at least one reward field to search.")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("each reward field may be selected once.")


def spec_to_json(spec: SearchSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "game": spec.game,
        "parent": spec.parent,
        "checkpoint": spec.checkpoint,
        "fields": list(spec.fields),
        "settings": settings_to_json(spec.settings),
    }


def spec_from_json(data: dict[str, Any]) -> SearchSpec:
    """``KeyError`` / ``TypeError`` / ``ValueError`` for anything that is not a spec."""
    return SearchSpec(
        name=str(data["name"]),
        game=str(data["game"]),
        parent=str(data["parent"]),
        checkpoint=str(data["checkpoint"]),
        fields=tuple(str(name) for name in data["fields"]),
        settings=settings_from_json(data["settings"]),
    )
