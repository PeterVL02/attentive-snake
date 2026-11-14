"""Reward schemes of a game (``configs/<game>/rewards/*.toml``): list, read, save, delete, and preview what a scheme pays.

Every route takes ``?game=`` (default ``snake``).
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from attentive_snake.app.config_json import dataclass_to_json
from attentive_snake.app.errors import invalid_input_as_422, not_found
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam, known_game
from attentive_snake.app.paths import AppPaths
from attentive_snake.config.dataclass_table import dataclass_from_table
from attentive_snake.config.layout import DEFAULT_SCHEME, rewards_dir
from attentive_snake.config.reward_schemes import delete_scheme, list_schemes, load_scheme, save_scheme
from attentive_snake.games.spec import GameSpec

PROTECTED_SCHEMES = frozenset({DEFAULT_SCHEME})


class Preview(BaseModel):
    reward: dict[str, Any]
    example_score: int = Field(default=5, ge=0, le=10_000)


def _reward_config(game: GameSpec, fields: dict[str, Any]) -> Any:
    with invalid_input_as_422():
        return dataclass_from_table(game.reward_type, fields, section="reward")


def make_router(paths: AppPaths) -> APIRouter:
    router = APIRouter(prefix="/api/rewards")

    def directory(game: GameSpec) -> Path:
        return rewards_dir(paths.configs_dir, game.name)

    def existing(game: GameSpec, name: str) -> str:
        with invalid_input_as_422():
            names = list_schemes(directory(game))
        if name not in names:
            raise not_found(f"no {game.name} reward scheme named {name!r}.")
        return name

    @router.get("")
    def list_all(game: GameParam = DEFAULT_GAME) -> dict:
        spec = known_game(game)
        return {"game": spec.name, "schemes": list_schemes(directory(spec)), "protected": sorted(PROTECTED_SCHEMES)}

    @router.post("/preview")
    def preview(request: Preview, game: GameParam = DEFAULT_GAME) -> list[dict]:
        spec = known_game(game)
        if spec.reward_table is None:
            raise not_found(f"the {spec.name} game has no reward preview.")
        return spec.reward_table(_reward_config(spec, request.reward), request.example_score)

    @router.get("/{name}")
    def read(name: str, game: GameParam = DEFAULT_GAME) -> dict:
        spec = known_game(game)
        with invalid_input_as_422():
            return dataclass_to_json(load_scheme(directory(spec), existing(spec, name), spec.reward_type))

    @router.put("/{name}")
    def save(name: str, fields: dict[str, Any], game: GameParam = DEFAULT_GAME) -> dict:
        spec = known_game(game)
        config = _reward_config(spec, fields)
        with invalid_input_as_422():  # the scheme name is validated by save_scheme
            save_scheme(directory(spec), name, config)
        return {"name": name, "game": spec.name, "reward": dataclass_to_json(config)}

    @router.delete("/{name}", status_code=204)
    def delete(name: str, game: GameParam = DEFAULT_GAME) -> None:
        spec = known_game(game)
        if name in PROTECTED_SCHEMES:
            raise HTTPException(status_code=403, detail=f"the {name!r} scheme cannot be deleted.")
        with invalid_input_as_422():
            delete_scheme(directory(spec), existing(spec, name))

    return router
