"""The request body of routes that look at a not yet started config: its experiment tables, in the form's game."""

from typing import Any

from pydantic import BaseModel, Field

from attentive_snake.app.game_lookup import known_game
from attentive_snake.app.launch import parse_experiment
from attentive_snake.config import ExperimentConfig


class ConfigRequest(BaseModel):
    config: dict[str, dict[str, Any]] = Field(description="experiment tables, as when starting a run; [game] name is taken from ?game=")


def experiment_of(request: ConfigRequest, game: str) -> ExperimentConfig:
    """The config of ``request`` for ``game`` (404 for an unknown game, 422 for an invalid config)."""
    spec = known_game(game)
    tables = {**request.config, "game": {**request.config.get("game", {}), "name": spec.name}}
    return parse_experiment(tables)
