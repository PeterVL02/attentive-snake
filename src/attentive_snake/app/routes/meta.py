"""Static information for the UI: the config schema, the default experiment, the fork rule and the CLI cheat sheet."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.errors import not_found
from attentive_snake.app.paths import AppPaths
from attentive_snake.config import load_experiment
from attentive_snake.config.fork_rules import locked_fields
from attentive_snake.config.schema import experiment_schema


def make_router(paths: AppPaths) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/schema")
    def schema() -> dict:
        return experiment_schema()

    @router.get("/defaults")
    def defaults() -> dict:
        return experiment_to_json(load_experiment(paths.default_config))

    @router.get("/fork-rules")
    def fork_rules() -> dict:
        """The settings a run continued under a new name must keep from its parent, with the reason."""
        return {"locked": locked_fields()}

    @router.get("/cheatsheet", response_class=PlainTextResponse)
    def cheat_sheet() -> str:
        if not paths.cheat_sheet.is_file():
            raise not_found(f"{paths.cheat_sheet} does not exist.")
        return paths.cheat_sheet.read_text(encoding="utf-8")

    return router
