"""Resolving the ``game`` request parameter, and the facts about a game the UI needs."""

from typing import Annotated

from fastapi import Query

from dataclasses import asdict

from attentive_snake.app.errors import not_found
from attentive_snake.evaluation.statistics import metric_definitions
from attentive_snake.games.registry import DEFAULT_GAME, get_game
from attentive_snake.games.spec import GameSpec

# Use as ``game: GameParam = DEFAULT_GAME`` in a route's signature (a ``?game=`` query parameter).
type GameParam = Annotated[str, Query(description="name of the game (see /api/games)")]
__all__ = ["DEFAULT_GAME", "GameParam", "game_info", "known_game"]


def known_game(name: str) -> GameSpec:
    """The game called ``name``; 404 listing the known games otherwise (so names never reach the file system unchecked)."""
    try:
        return get_game(name)
    except ValueError as error:
        raise not_found(str(error)) from error


def game_info(game: GameSpec) -> dict:
    return {
        "name": game.name,
        "title": game.title,
        "description": game.description,
        "observation_kind": game.observation_kind,
        "view_kind": game.view_kind,
        "has_reward_table": game.reward_table is not None,
        "evaluation_metrics": [asdict(definition) for definition in metric_definitions(game)],
    }
