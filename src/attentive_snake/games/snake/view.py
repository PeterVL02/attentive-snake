"""A Snake ``GameState`` as the ``"board"`` view payload (see ``attentive_snake.views.board``)."""

from typing import Any

from attentive_snake.games.snake.types import GameState
from attentive_snake.views.board import board_payload


def board_view(state: GameState) -> dict[str, Any]:
    return board_payload(state.width, state.height, state.snake, state.food)
