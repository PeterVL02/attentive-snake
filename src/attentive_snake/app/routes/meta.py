"""Static information for the UI: the games, the config schema and default experiment of a game (per algorithm), the fork rule and the CLI cheat sheet."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.errors import not_found
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam, game_info, known_game
from attentive_snake.app.paths import AppPaths
from attentive_snake.config.defaults import default_experiment
from attentive_snake.config.fork_rules import locked_fields
from attentive_snake.config.layout import default_config_path
from attentive_snake.config.schema import experiment_schema
from attentive_snake.training.algorithms import get_algorithm
from attentive_snake.games.registry import game_names


def make_router(paths: AppPaths) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/games")
    def games() -> list[dict]:
        """Every game that can be trained, alphabetical."""
        return [game_info(known_game(name)) for name in game_names()]

    @router.get("/schema")
    def schema(game: GameParam = DEFAULT_GAME) -> dict:
        return experiment_schema(known_game(game).name)

    @router.get("/defaults")
    def defaults(game: GameParam = DEFAULT_GAME, algorithm: str | None = None) -> dict:
        """The game's default experiment; with ``algorithm``, its defaults for that algorithm (404 if unknown)."""
        name = known_game(game).name
        path = default_config_path(paths.configs_dir, name)
        if not path.is_file():
            raise not_found(f"{path} does not exist.")
        if algorithm is not None:
            try:
                get_algorithm(algorithm)
            except ValueError as error:
                raise not_found(str(error)) from error
        return experiment_to_json(default_experiment(paths.configs_dir, name, algorithm))

    @router.get("/fork-rules")
    def fork_rules(game: GameParam = DEFAULT_GAME) -> dict:
        """The settings a run continued under a new name must keep from its parent, with the reason."""
        return {"locked": locked_fields(known_game(game).name)}

    @router.get("/cheatsheet", response_class=PlainTextResponse)
    def cheat_sheet() -> str:
        if not paths.cheat_sheet.is_file():
            raise not_found(f"{paths.cheat_sheet} does not exist.")
        return paths.cheat_sheet.read_text(encoding="utf-8")

    return router
