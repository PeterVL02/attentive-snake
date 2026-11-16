"""Request bodies of the routes that start or queue training, and the launch request each one describes."""

from typing import Any

from pydantic import BaseModel, Field

from attentive_snake.app import launch_request
from attentive_snake.app.launch import parse_experiment
from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.run_metadata import clean_description

DESCRIPTION_LENGTH = 2000


class NewRun(BaseModel):
    name: str
    config: dict[str, dict[str, Any]] = Field(description="experiment tables: {section: {field: value}}; game = config.game.name")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")

    def launch(self) -> LaunchRequest:
        """The request for this new run; its game is the config's (422 if the config is invalid)."""
        game = parse_experiment(self.config).game_name
        return launch_request.new_run(game, self.name, self.config, clean_description(self.description))


class Resume(BaseModel):
    total_env_steps: int | None = Field(default=None, ge=1, description="train until this many env steps")
    keep_checkpoints: bool | None = Field(default=None, description="change train.keep_checkpoints (None: keep it)")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")
    evaluation_interval: int | None = Field(default=None, ge=0, description="change evaluation.interval (0: off; None: keep it)")
    evaluation_games: int | None = Field(default=None, ge=1, description="change evaluation.games (None: keep it)")
    evaluation_seed: int | None = Field(default=None, description="change evaluation.seed (None: keep it)")

    def evaluation_changes(self) -> dict[str, int]:
        """The changed ``[evaluation]`` settings, ``{"interval" | "games" | "seed": value}``."""
        values = {"interval": self.evaluation_interval, "games": self.evaluation_games, "seed": self.evaluation_seed}
        return {key: value for key, value in values.items() if value is not None}

    def launch(self, game: str, run: str) -> LaunchRequest:
        return launch_request.resume(
            game, run, self.total_env_steps, self.keep_checkpoints, clean_description(self.description), self.evaluation_changes()
        )


class Fork(BaseModel):
    name: str = Field(description="name of the new run (of the same game)")
    checkpoint: str | None = Field(default=None, description="checkpoint file of the parent (None: its latest)")
    config: dict[str, dict[str, Any]] = Field(description="experiment tables of the new run")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")

    def launch(self, game: str, parent: str) -> LaunchRequest:
        return launch_request.fork(game, parent, self.name, self.config, self.checkpoint, clean_description(self.description))
