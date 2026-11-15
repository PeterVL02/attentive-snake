"""How many trainable parameters the policy of a not yet started config would have."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from attentive_snake.app.errors import invalid_input_as_422
from attentive_snake.app.game_lookup import GameParam, known_game
from attentive_snake.app.launch import parse_experiment
from attentive_snake.app.parameter_count import config_parameter_count


class ParameterRequest(BaseModel):
    config: dict[str, dict[str, Any]] = Field(description="experiment tables, as when starting a run; [game] name is taken from ?game=")


def make_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/parameters")
    def count(request: ParameterRequest, game: GameParam) -> dict:
        spec = known_game(game)
        tables = {**request.config, "game": {**request.config.get("game", {}), "name": spec.name}}
        config = parse_experiment(tables)
        with invalid_input_as_422():  # e.g. a grid too small for the model
            return {"game": spec.name, "parameters": config_parameter_count(config)}

    return router
