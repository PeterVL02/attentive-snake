"""The FastAPI application: wires the routers, the static frontend and the training-job lifecycle."""

import atexit
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from attentive_snake.app.background import BackgroundJobs
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.paths import STATIC_DIR, AppPaths, default_paths
from attentive_snake.app.queue_storage import QueueFile, queue_path
from attentive_snake.app.queue_ticker import Ticker
from attentive_snake.app.routes import action_space, architecture, compare, evaluations, meta, parameters, queue, reward_search, rewards, runs, training_warnings, watch
from attentive_snake.app.search_backend import QueueBackend
from attentive_snake.app.training_launcher import TrainingLauncher
from attentive_snake.app.training_queue import TrainingQueue
from attentive_snake.app.watch_sessions import WatchSessions
from attentive_snake.reward_search.searches import RewardSearches
from attentive_snake.reward_search.storage import SearchFiles, searches_dir


def create_app(paths: AppPaths | None = None, jobs: TrainingJobs | None = None, queue_interval: float = 1.0) -> FastAPI:
    """The app; training jobs still running when it shuts down are stopped.

    The training queue (``<runs root>/queue.json``) and the running reward searches
    (``<runs root>/reward_searches/``) are looked at every ``queue_interval`` seconds while the app
    runs; what still waits in the queue when the app stops starts with the next app, and a search
    carries on where it was.
    """
    paths = paths or default_paths()
    jobs = jobs or TrainingJobs(paths.working_dir)
    launcher = TrainingLauncher(paths, jobs, TrainingQueue(QueueFile(queue_path(paths.runs_dir))))
    ticker = Ticker(launcher.runner.tick, queue_interval)
    searches = RewardSearches(SearchFiles(searches_dir(paths.runs_dir)), QueueBackend(paths, jobs, launcher))
    search_ticker = Ticker(searches.tick, queue_interval, name="reward search")

    def shutdown() -> None:
        search_ticker.stop()  # first: a job stopped below must not count as a member that ended
        ticker.stop()  # before the jobs: a stopped job must not make the queue start the next one
        jobs.shutdown()

    atexit.register(shutdown)  # also covers exits that skip the lifespan shutdown

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        ticker.start()
        search_ticker.start()
        yield
        shutdown()

    app = FastAPI(title="Attentive Snake", lifespan=lifespan)
    app.state.paths = paths
    app.state.jobs = jobs
    app.state.launcher = launcher  # start or queue training without HTTP (see training_launcher)
    app.state.searches = searches  # reward searches without HTTP (see reward_search.searches)
    app.include_router(meta.make_router(paths))
    app.include_router(runs.make_router(paths, jobs, launcher))
    app.include_router(queue.make_router(launcher))
    app.include_router(reward_search.make_router(paths, searches))
    app.include_router(evaluations.make_router(paths, BackgroundJobs()))
    app.include_router(architecture.make_router(paths))
    app.include_router(parameters.make_router())
    app.include_router(training_warnings.make_router())
    app.include_router(action_space.make_router())
    app.include_router(compare.make_router(paths, jobs))
    app.include_router(rewards.make_router(paths))
    app.include_router(watch.make_router(paths, WatchSessions()))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
