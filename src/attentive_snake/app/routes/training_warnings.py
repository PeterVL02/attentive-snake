"""The game's plain-language warnings about a config that is about to be trained."""

from fastapi import APIRouter

from attentive_snake.app.config_request import ConfigRequest, experiment_of
from attentive_snake.app.game_lookup import GameParam
from attentive_snake.training_warnings import training_warnings


def make_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/training-warnings")
    def warnings(request: ConfigRequest, game: GameParam) -> dict:
        return {"warnings": list(training_warnings(experiment_of(request, game)))}

    return router
