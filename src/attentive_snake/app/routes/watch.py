"""Watch sessions: a trained policy playing one game at a time, stepped by the browser."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from attentive_snake.app.errors import invalid_input_as_422, not_found
from attentive_snake.app.game_lookup import DEFAULT_GAME
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import checkpoint_in_run, existing_run_dir
from attentive_snake.app.watch_sessions import UnknownSession, WatchSessions
from attentive_snake.watch import WatchFrame, WatchSession

MAX_STEPS_PER_CALL = 100
EPISODE_KEYS = ("score", "steps", "event", "done", "episode")  # what a frame that is not drawn still needs


class Options(BaseModel):
    sample: bool | None = None
    attention: bool | None = None
    layer: int | None = None
    frame: int | None = None


class NewSession(BaseModel):
    game: str = Field(default=DEFAULT_GAME, description="the game of the run")
    run: str
    checkpoint: str | None = Field(default=None, description="checkpoint file name; default: the latest")
    sample: bool = False
    attention: bool = True
    layer: int = -1
    frame: int = -1
    seed: int | None = None


class Steps(BaseModel):
    count: int = Field(default=1, ge=1, le=MAX_STEPS_PER_CALL)
    frames: Literal["all", "last"] = Field(
        default="all",
        description='"last": only the last frame in full; the others carry only the episode keys (score, steps, event, done, episode)',
    )


def step_frames(frames: list[WatchFrame], keep: Literal["all", "last"]) -> list[dict[str, Any]]:
    """The frames as JSON; with ``keep="last"`` all but the last are cut down to ``EPISODE_KEYS`` (no view, no attention)."""
    dicts = [frame.to_dict() for frame in frames]
    if keep == "last":
        dicts[:-1] = [{key: frame[key] for key in EPISODE_KEYS} for frame in dicts[:-1]]
    return dicts


def make_router(paths: AppPaths, sessions: WatchSessions) -> APIRouter:
    router = APIRouter(prefix="/api/watch")

    @contextmanager
    def session(session_id: str) -> Iterator[Any]:
        try:
            with sessions.use(session_id) as found:
                yield found
        except UnknownSession:
            raise not_found(f"no watch session {session_id!r} (it may have been closed to make room).") from None

    @router.post("", status_code=201)
    def create(request: NewSession) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, request.game, request.run)
        checkpoint = None if request.checkpoint is None else checkpoint_in_run(run_dir, request.checkpoint)
        try:
            with invalid_input_as_422():
                watch = WatchSession(
                    run_dir,
                    checkpoint=checkpoint,
                    sample=request.sample,
                    attention=request.attention,
                    layer=request.layer,
                    frame=request.frame,
                    seed=request.seed,
                )
        except FileNotFoundError as error:  # the run has no checkpoint yet
            raise not_found(str(error)) from error
        return {"id": sessions.add(watch), "info": watch.info, "frame": watch.current().to_dict()}

    @router.post("/{session_id}/step")
    def step(session_id: str, request: Steps) -> dict:
        with session(session_id) as watch:
            return {"frames": step_frames([watch.step() for _ in range(request.count)], request.frames)}

    @router.post("/{session_id}/reset")
    def reset(session_id: str) -> dict:
        with session(session_id) as watch:
            return {"frame": watch.reset().to_dict()}

    @router.post("/{session_id}/options")
    def options(session_id: str, request: Options) -> dict:
        with session(session_id) as watch, invalid_input_as_422():
            watch.set_options(**request.model_dump(exclude_none=True))
            return {"info": watch.info, "frame": watch.current().to_dict()}

    @router.delete("/{session_id}", status_code=204)
    def delete(session_id: str) -> None:
        try:
            sessions.remove(session_id)
        except UnknownSession:
            raise not_found(f"no watch session {session_id!r}.") from None

    return router
