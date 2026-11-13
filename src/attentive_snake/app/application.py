"""The FastAPI application: wires the routers, the static frontend and the training-job lifecycle."""

import atexit
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.paths import STATIC_DIR, AppPaths, default_paths
from attentive_snake.app.routes import architecture, compare, meta, rewards, runs, watch
from attentive_snake.app.watch_sessions import WatchSessions


def create_app(paths: AppPaths | None = None, jobs: TrainingJobs | None = None) -> FastAPI:
    """The app; training jobs still running when it shuts down are stopped."""
    paths = paths or default_paths()
    jobs = jobs or TrainingJobs(paths.working_dir)
    atexit.register(jobs.shutdown)  # also covers exits that skip the lifespan shutdown

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.shutdown()

    app = FastAPI(title="Attentive Snake", lifespan=lifespan)
    app.state.paths = paths
    app.state.jobs = jobs
    app.include_router(meta.make_router(paths))
    app.include_router(runs.make_router(paths, jobs))
    app.include_router(architecture.make_router(paths))
    app.include_router(compare.make_router(paths, jobs))
    app.include_router(rewards.make_router(paths))
    app.include_router(watch.make_router(paths, WatchSessions()))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
