"""Reward searches: the start form's rows, start one from a run's checkpoint, list them, read one, stop one.

``GET /api/reward-search/form?game=``, ``POST /api/reward-search?game=`` (201),
``GET /api/reward-search[?game=]``, ``GET /api/reward-search/{name}``, ``POST /api/reward-search/{name}/stop``.
Member runs are ordinary runs of the game, queued in the training queue (see ``search_backend``).
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from attentive_snake.app.errors import conflict, invalid_input_as_422, not_found
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam, known_game
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import checkpoint_in_run, existing_run_dir, latest_checkpoint_in_run
from attentive_snake.app.search_info import search_detail, search_form, search_summary
from attentive_snake.reward_search.searches import RewardSearches, SearchExists
from attentive_snake.reward_search.settings import settings_from_json
from attentive_snake.reward_search.spec import SearchSpec

RESERVED_NAMES = frozenset({"form"})  # GET /api/reward-search/form is the form, not a search


class StartSearch(BaseModel):
    name: str = Field(description="name of the search; member runs are called <name>-g<generation>-m<member>")
    parent: str = Field(description="the run to start from (of the game in ?game=)")
    checkpoint: str | None = Field(default=None, description="the parent's checkpoint file (None: its latest)")
    fields: list[str] = Field(description="the [reward] fields the search may change")
    settings: dict[str, Any] = Field(default_factory=dict, description="SearchSettings values; missing ones take their defaults")


def make_router(paths: AppPaths, searches: RewardSearches) -> APIRouter:
    router = APIRouter(prefix="/api/reward-search")

    def existing(name: str) -> Any:
        try:
            return searches.get(name)
        except LookupError as error:
            raise not_found(str(error)) from error
        except ValueError as error:  # a bad name, or a file that cannot be read
            raise not_found(str(error)) from error

    @router.get("/form")
    def form(game: GameParam = DEFAULT_GAME) -> dict:
        return search_form(known_game(game))

    @router.get("")
    def list_all(game: str | None = None) -> list[dict]:
        name = None if game is None else known_game(game).name
        return [search_summary(state) for state in searches.all() if name in (None, state.spec.game)]

    @router.post("", status_code=201)
    def start(request: StartSearch, game: GameParam = DEFAULT_GAME) -> dict:
        spec_game = known_game(game)
        if request.name in RESERVED_NAMES:
            raise HTTPException(status_code=422, detail=f"{request.name!r} cannot name a search (it is a route of its own); choose another name.")
        parent_dir = existing_run_dir(paths.runs_dir, spec_game.name, request.parent)
        checkpoint = latest_checkpoint_in_run(parent_dir) if request.checkpoint is None else checkpoint_in_run(parent_dir, request.checkpoint)
        with invalid_input_as_422():
            spec = SearchSpec(request.name, spec_game.name, parent_dir.name, checkpoint.name, tuple(request.fields), settings_from_json(request.settings))
            try:
                return search_detail(searches.start(spec))
            except SearchExists as error:
                raise conflict(str(error)) from error

    @router.get("/{name}")
    def detail(name: str) -> dict:
        return search_detail(existing(name))

    @router.post("/{name}/stop")
    def stop(name: str) -> dict:
        existing(name)
        return search_detail(searches.stop(name))

    return router
