"""How many trainable parameters the policy of a not yet started config would have."""

from fastapi import APIRouter

from attentive_snake.app.config_request import ConfigRequest, experiment_of
from attentive_snake.app.errors import invalid_input_as_422
from attentive_snake.app.game_lookup import GameParam
from attentive_snake.app.parameter_count import config_parameter_count


def make_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/parameters")
    def count(request: ConfigRequest, game: GameParam) -> dict:
        config = experiment_of(request, game)
        with invalid_input_as_422():  # e.g. a grid too small for the model
            return {"game": config.game_name, "parameters": config_parameter_count(config)}

    return router
