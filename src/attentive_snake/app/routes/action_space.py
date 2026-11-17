"""Whether a game's rules give continuous actions, so the New run form never presets a setting that needs discrete ones."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from attentive_snake.app.errors import invalid_input_as_422
from attentive_snake.app.game_lookup import GameParam, known_game
from attentive_snake.config.dataclass_table import dataclass_from_table


class RulesRequest(BaseModel):
    rules: dict[str, Any] = Field(default_factory=dict, description="the [game] table without name; absent keys keep the game's defaults")


def make_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/action-space")
    def action_space(request: RulesRequest, game: GameParam) -> dict:
        spec = known_game(game)
        with invalid_input_as_422():
            rules = dataclass_from_table(spec.rules_type, request.rules, "game")
        return {"continuous": spec.has_continuous_actions(rules)}

    return router
