"""Starting a launch request now or putting it in the training queue: the entry point of the routes and of in-process drivers.

Both ways validate the request the same way (``launch_plan``) and respect the names the queue
has reserved, so a queued request does not fail later for a reason that was known when it was
queued. Without HTTP::

    entry = launcher.enqueue(launch_request.fork("snake", "parent", "child", tables))   # HTTPException 404/409/422
    outcome = launcher.outcome(entry.id)        # .state, .done, .run_dir once started, .error

``queue.max_parallel`` jobs started by the queue run at once (``launcher.queue.set_max_parallel``).
"""

from attentive_snake.app.errors import conflict
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch_plan import LaunchedRun, plan_launch, start_launch
from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.queue_entry import QueueEntry
from attentive_snake.app.queue_runner import EntryOutcome, QueueRunner
from attentive_snake.app.training_queue import NameReserved, TrainingQueue


class TrainingLauncher:
    def __init__(self, paths: AppPaths, jobs: TrainingJobs, queue: TrainingQueue | None = None) -> None:
        """``queue``: default a queue kept in memory only."""
        self._paths = paths
        self._jobs = jobs
        self.queue = queue or TrainingQueue()
        self.runner = QueueRunner(self.queue, jobs, self._launch)

    def start(self, request: LaunchRequest) -> LaunchedRun:
        """Start ``request``'s job now; 409 if its name is reserved by a queued entry, else as ``start_launch``."""
        if request.kind != "resume" and self.queue.is_reserved(request.game, request.run):
            raise conflict(f"run {request.run!r} is waiting in the training queue; choose another name, or remove it there.")
        return self._launch(request)

    def enqueue(self, request: LaunchRequest) -> QueueEntry:
        """Validate ``request`` as a start would and append it to the queue; it starts at once if a slot is free.

        404/409/422 as ``plan_launch``; 409 also if an entry for the same run is already waiting.
        """
        plan_launch(self._paths, request)
        try:
            entry = self.queue.add(request)
        except NameReserved as error:
            raise conflict(str(error)) from error
        self.runner.tick()
        return entry

    def outcome(self, entry_id: str) -> EntryOutcome:
        """What became of a queued entry (see ``EntryOutcome``)."""
        return self.runner.outcome(entry_id)

    def _launch(self, request: LaunchRequest) -> LaunchedRun:
        return start_launch(self._paths, self._jobs, request)
