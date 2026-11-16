"""The training queue: list it, add a new run / fork / resume, start an entry now, remove one, and its ``max_parallel``.

There is one queue for all games (one machine trains them all); every entry carries its game.
Adding mirrors the routes that start a run at once (``routes/runs.py``), with the same bodies and
the same ``?game=`` (default ``snake``; a new run's game is its config's ``[game] name``):
``POST /api/queue/runs``, ``POST /api/queue/runs/{name}/fork``, ``POST /api/queue/runs/{name}/resume``.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from attentive_snake.app.errors import not_found
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam, known_game
from attentive_snake.app.launch_bodies import Fork, NewRun, Resume
from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.app.queue_info import entry_info
from attentive_snake.app.training_launcher import TrainingLauncher


class QueueSettings(BaseModel):
    max_parallel: int = Field(ge=0, le=64, description="queue-started jobs running at once (0: the queue is paused)")


def make_router(launcher: TrainingLauncher) -> APIRouter:
    router = APIRouter(prefix="/api/queue")
    queue, runner = launcher.queue, launcher.runner

    def settings() -> dict:
        return {"max_parallel": queue.max_parallel}

    def queued(request: LaunchRequest) -> dict:
        """Add ``request`` to the queue and describe its entry; ``outcome`` says whether it waits or started at once."""
        entry = launcher.enqueue(request)
        return {**entry_info(entry), "outcome": launcher.outcome(entry.id).state}

    @router.get("")
    def list_all(game: str | None = None) -> dict:
        """The whole queue in order, or only ``game``'s entries; ``running``: entries started from it that still train."""
        name = None if game is None else known_game(game).name
        entries = [entry for entry in queue.entries() if name in (None, entry.request.game)]
        running = [entry for entry in runner.running() if name in (None, entry.request.game)]
        return {
            **settings(),
            "entries": [entry_info(entry) for entry in entries],
            "running": [{"id": entry.id, "game": entry.request.game, "run": entry.request.run} for entry in running],
        }

    @router.post("/runs", status_code=201)
    def add_new(request: NewRun) -> dict:
        return queued(request.launch())

    @router.post("/runs/{name}/fork", status_code=201)
    def add_fork(name: str, request: Fork, game: GameParam = DEFAULT_GAME) -> dict:
        return queued(request.launch(game, name))

    @router.post("/runs/{name}/resume", status_code=201)
    def add_resume(name: str, request: Resume, game: GameParam = DEFAULT_GAME) -> dict:
        return queued(request.launch(game, name))

    @router.get("/settings")
    def get_settings() -> dict:
        return settings()

    @router.put("/settings")
    def set_settings(request: QueueSettings) -> dict:
        queue.set_max_parallel(request.max_parallel)
        runner.tick()
        return settings()

    @router.post("/{entry_id}/run")
    def run_now(entry_id: str) -> dict:
        """Start the entry at once, whatever is running. If it cannot start, it is kept as failed and the error returned."""
        try:
            launched = runner.run_now(entry_id)
        except LookupError as error:
            raise not_found(str(error)) from error
        return {"name": launched.run_dir.name, "game": launched.run_dir.parent.name, "status": launched.status}

    @router.delete("/{entry_id}")
    def remove(entry_id: str) -> dict:
        try:
            return entry_info(queue.remove(entry_id))
        except LookupError as error:
            raise not_found(str(error)) from error

    return router
