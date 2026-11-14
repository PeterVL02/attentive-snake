"""Architecture diagrams for a posted config (its game is ``[game] name``) or for an existing run (``?game=``)."""

from typing import Any

from fastapi import APIRouter

from attentive_snake.app.architecture import describe_architecture
from attentive_snake.app.errors import invalid_input_as_422
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam
from attentive_snake.app.launch import parse_experiment
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import existing_run_dir
from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path


def make_router(paths: AppPaths) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/architecture")
    def for_config(tables: dict[str, dict[str, Any]]) -> dict:
        config = parse_experiment(tables)
        with invalid_input_as_422():  # e.g. a grid too small for the model
            return describe_architecture(config)

    @router.get("/runs/{name}/architecture")
    def for_run(name: str, game: GameParam = DEFAULT_GAME) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        return describe_architecture(load_experiment(config_path(run_dir)))

    return router
