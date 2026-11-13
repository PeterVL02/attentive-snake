"""Reward schemes (``configs/rewards/*.toml``): list, read, save, delete, and preview what a scheme pays."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from attentive_snake.app.config_json import dataclass_to_json
from attentive_snake.app.errors import invalid_input_as_422, not_found
from attentive_snake.app.paths import AppPaths
from attentive_snake.config.dataclass_table import dataclass_from_table
from attentive_snake.config.reward_schemes import (
    delete_scheme,
    list_schemes,
    load_scheme,
    reward_table,
    save_scheme,
)
from attentive_snake.env.reward import RewardConfig

PROTECTED_SCHEMES = frozenset({"default"})


class Preview(BaseModel):
    reward: dict[str, Any]
    example_score: int = Field(default=5, ge=0, le=10_000)


def _reward_config(fields: dict[str, Any]) -> RewardConfig:
    with invalid_input_as_422():
        return dataclass_from_table(RewardConfig, fields, section="reward")


def make_router(paths: AppPaths) -> APIRouter:
    router = APIRouter(prefix="/api/rewards")

    def existing(name: str) -> str:
        with invalid_input_as_422():
            names = list_schemes(paths.rewards_dir)
        if name not in names:
            raise not_found(f"no reward scheme named {name!r}.")
        return name

    @router.get("")
    def list_all() -> dict:
        return {"schemes": list_schemes(paths.rewards_dir), "protected": sorted(PROTECTED_SCHEMES)}

    @router.post("/preview")
    def preview(request: Preview) -> list[dict]:
        return reward_table(_reward_config(request.reward), example_score=request.example_score)

    @router.get("/{name}")
    def read(name: str) -> dict:
        with invalid_input_as_422():
            return dataclass_to_json(load_scheme(paths.rewards_dir, existing(name)))

    @router.put("/{name}")
    def save(name: str, fields: dict[str, Any]) -> dict:
        config = _reward_config(fields)
        with invalid_input_as_422():  # the scheme name is validated by save_scheme
            save_scheme(paths.rewards_dir, name, config)
        return {"name": name, "reward": dataclass_to_json(config)}

    @router.delete("/{name}", status_code=204)
    def delete(name: str) -> None:
        if name in PROTECTED_SCHEMES:
            raise HTTPException(status_code=403, detail=f"the {name!r} scheme cannot be deleted.")
        with invalid_input_as_422():
            delete_scheme(paths.rewards_dir, existing(name))

    return router
