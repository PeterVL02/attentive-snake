"""The training queue without the app: the queue itself, its file, the runner that starts entries and the ticker."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from attentive_snake.app import launch_request
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch_plan import LaunchedRun
from attentive_snake.app.launch_request import LaunchRequest, request_from_json, request_to_json
from attentive_snake.app.queue_entry import entry_from_json, entry_to_json, new_entry
from attentive_snake.app.queue_runner import QueueRunner
from attentive_snake.app.queue_storage import QueueFile, QueueState, queue_path
from attentive_snake.app.queue_ticker import Ticker
from attentive_snake.app.run_lookup import run_job_key
from attentive_snake.app.training_queue import NameReserved, TrainingQueue

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
QUICK = [sys.executable, "-c", "pass"]


def request(run: str, game: str = "snake") -> LaunchRequest:
    return launch_request.new_run(game, run, {"game": {"name": game}})


def wait_for(condition, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


# --------------------------------------------------------------------------- requests and entries


def test_requests_and_entries_round_trip_through_json():
    requests = [
        launch_request.new_run("snake", "a", {"train": {"seed": 1}}, "note"),
        launch_request.fork("snake", "parent", "child", {"reward": {"food": 2.0}}, "checkpoint_0000000016.pt"),
        launch_request.resume("walk", "old", 128, False, None, {"interval": 0}),
    ]
    assert [r.kind for r in requests] == ["new", "fork", "resume"]
    for original in requests:
        assert request_from_json(json.loads(json.dumps(request_to_json(original)))) == original
        entry = new_entry(original)
        assert entry_from_json(json.loads(json.dumps(entry_to_json(entry)))) == entry
    assert len({new_entry(requests[0]).id for _ in range(50)}) == 50


# --------------------------------------------------------------------------- the queue


def test_queue_keeps_order_reserves_names_and_removes():
    queue = TrainingQueue()
    a, b = queue.add(request("a")), queue.add(request("b"))
    other_game = queue.add(request("a", game="walk"))  # names are per game
    assert [entry.id for entry in queue.entries()] == [a.id, b.id, other_game.id]
    assert queue.is_reserved("snake", "a") and queue.is_reserved("walk", "a") and not queue.is_reserved("walk", "b")
    with pytest.raises(NameReserved):
        queue.add(request("a"))
    assert queue.remove(a.id) == a
    with pytest.raises(LookupError):
        queue.remove(a.id)
    assert not queue.is_reserved("snake", "a")
    queue.add(request("a"))
    assert [entry.request.run for entry in queue.entries()] == ["b", "a", "a"]


def test_a_claimed_entry_starts_once_and_fails_in_place():
    queue = TrainingQueue()
    a, b, c = (queue.add(request(name)) for name in "abc")
    claimed = queue.claim_next()
    assert claimed.id == a.id and claimed.state == "starting"
    assert queue.is_reserved("snake", "a")  # still reserved while it starts
    for refused in (queue.claim, queue.remove):
        with pytest.raises(LookupError):
            refused(a.id)
    assert queue.claim_next().id == b.id
    queue.discard(a.id)
    queue.mark_failed(b.id, "no checkpoint")
    assert [(e.request.run, e.state, e.error) for e in queue.entries()] == [("b", "failed", "no checkpoint"), ("c", "queued", None)]
    assert not queue.is_reserved("snake", "b")  # a failed entry does not hold its name
    assert queue.claim_next().id == c.id and queue.claim_next() is None  # a failed entry is not started by itself
    retried = queue.claim(b.id)  # ... but may be started on request
    assert retried.state == "starting" and retried.error is None


def test_max_parallel():
    queue = TrainingQueue()
    assert queue.max_parallel == 1
    queue.set_max_parallel(0)
    queue.set_max_parallel(3)
    assert queue.max_parallel == 3
    with pytest.raises(ValueError):
        queue.set_max_parallel(-1)


def test_concurrent_claims_give_every_entry_to_exactly_one_thread():
    queue = TrainingQueue()
    ids = [queue.add(request(f"run-{i}")).id for i in range(200)]
    claimed: list[str] = []

    def worker() -> None:
        while (entry := queue.claim_next()) is not None:
            claimed.append(entry.id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(claimed) == sorted(ids)


# --------------------------------------------------------------------------- the file


def test_the_queue_file_restores_entries_order_and_max_parallel(tmp_path: Path):
    path = queue_path(tmp_path / "runs")
    queue = TrainingQueue(QueueFile(path))
    assert queue.entries() == [] and not path.exists()
    a, b, c = (queue.add(request(name)) for name in "abc")
    queue.set_max_parallel(2)
    queue.remove(b.id)
    queue.claim(a.id)  # the server dies while a is starting
    queue.mark_failed(queue.claim(c.id).id, "broken")
    assert json.loads(path.read_text())["version"] == 1
    assert [p.name for p in path.parent.iterdir()] == ["queue.json"]  # no temporary file left

    restored = TrainingQueue(QueueFile(path))
    assert restored.max_parallel == 2
    assert [(e.id, e.state, e.error) for e in restored.entries()] == [(a.id, "queued", None), (c.id, "failed", "broken")]
    assert restored.entries()[0].request == a.request


def test_an_unreadable_queue_file_is_moved_aside(tmp_path: Path):
    path = queue_path(tmp_path)
    path.write_text("{not json")
    assert QueueFile(path).load() == QueueState()
    assert not path.exists() and (tmp_path / "queue.json.unreadable").read_text() == "{not json"
    path.write_text(json.dumps({"version": 1, "max_parallel": 1, "entries": [{"id": "x"}]}))
    assert TrainingQueue(QueueFile(path)).entries() == []


# --------------------------------------------------------------------------- the runner


class Launches:
    """Stands in for ``start_launch``: a sleeping (or instantly ending) process per request; ``broken`` runs cannot start."""

    def __init__(self, jobs: TrainingJobs, root: Path) -> None:
        self.jobs = jobs
        self.root = root
        self.broken: set[str] = set()
        self.quick: set[str] = set()
        self.order: list[str] = []

    def __call__(self, launch: LaunchRequest) -> LaunchedRun:
        if launch.run in self.broken:
            raise RuntimeError(f"{launch.run} cannot start")
        run_dir = self.root / launch.game / launch.run
        command = QUICK if launch.run in self.quick else SLEEPER
        status = self.jobs.launch(run_job_key(run_dir), command, run_dir / "train.log", launch.kind)
        self.order.append(launch.run)
        return LaunchedRun(run_dir, status)


@pytest.fixture
def jobs(tmp_path: Path):
    training_jobs = TrainingJobs(tmp_path, stop_timeout=5)
    yield training_jobs
    training_jobs.shutdown()


@pytest.fixture
def launches(jobs: TrainingJobs, tmp_path: Path) -> Launches:
    return Launches(jobs, tmp_path / "runs")


def waiting(queue: TrainingQueue) -> list[str]:
    return [entry.request.run for entry in queue.entries() if entry.state == "queued"]


def test_runner_starts_one_after_another(jobs: TrainingJobs, launches: Launches):
    queue = TrainingQueue()
    runner = QueueRunner(queue, jobs, launches)
    entries = {name: queue.add(request(name)) for name in "abc"}
    assert runner.outcome(entries["a"].id).state == "queued" and not runner.outcome(entries["a"].id).done
    runner.tick()
    runner.tick()  # nothing more: the slot is taken
    assert launches.order == ["a"] and waiting(queue) == ["b", "c"]
    assert [entry.request.run for entry in runner.running()] == ["a"]
    outcome = runner.outcome(entries["a"].id)
    assert outcome.state == "running" and outcome.run_dir == launches.root / "snake" / "a" and not outcome.done

    jobs.stop("snake/a")
    runner.tick()
    assert launches.order == ["a", "b"] and waiting(queue) == ["c"]
    stopped = runner.outcome(entries["a"].id)
    assert stopped.state == "stopped" and stopped.done and stopped.run_dir == launches.root / "snake" / "a"
    assert runner.outcome("no-such-id").state == "unknown"


def test_runner_sees_a_job_end_by_itself_and_reports_how(jobs: TrainingJobs, launches: Launches):
    queue = TrainingQueue()
    runner = QueueRunner(queue, jobs, launches)
    launches.quick.add("a")
    a, b = queue.add(request("a")), queue.add(request("b"))
    runner.tick()
    wait_for(lambda: not jobs.is_running("snake/a"))
    assert runner.outcome(a.id).state == "finished" and runner.outcome(a.id).done
    runner.tick()
    assert runner.outcome(b.id).state == "running"
    jobs.launch("snake/a", SLEEPER, launches.root / "snake" / "a" / "train.log", "resume")  # the run is launched again
    assert runner.outcome(a.id).state == "ended" and runner.outcome(a.id).done  # only that it ended is known
    assert [entry.id for entry in runner.running()] == [b.id]  # the later job is not the queue's


def test_runner_runs_max_parallel_at_once_and_zero_pauses(jobs: TrainingJobs, launches: Launches):
    queue = TrainingQueue()
    runner = QueueRunner(queue, jobs, launches)
    queue.set_max_parallel(0)
    for name in "abcd":
        queue.add(request(name))
    runner.tick()
    assert launches.order == []
    queue.set_max_parallel(2)
    runner.tick()
    assert launches.order == ["a", "b"] and waiting(queue) == ["c", "d"]
    jobs.stop("snake/b")
    runner.tick()
    assert launches.order == ["a", "b", "c"]
    queue.set_max_parallel(1)  # lowering it stops nothing, it only keeps new ones from starting
    jobs.stop("snake/a")
    runner.tick()
    assert launches.order == ["a", "b", "c"] and waiting(queue) == ["d"]


def test_directly_started_jobs_and_run_now_take_no_slot(jobs: TrainingJobs, launches: Launches):
    queue = TrainingQueue()
    runner = QueueRunner(queue, jobs, launches)
    jobs.launch("snake/direct", SLEEPER, launches.root / "direct.log", "new")  # not started by the queue
    entries = {name: queue.add(request(name)) for name in "abc"}
    runner.tick()
    assert launches.order == ["a"]
    launched = runner.run_now(entries["c"].id)  # jumps the queue
    assert launched.run_dir.name == "c" and launched.status["state"] == "running"
    assert launches.order == ["a", "c"] and waiting(queue) == ["b"]
    assert sorted(entry.request.run for entry in runner.running()) == ["a", "c"]
    with pytest.raises(LookupError):
        runner.run_now(entries["c"].id)  # no longer in the queue
    jobs.stop("snake/a")
    runner.tick()  # c still runs, but it holds no slot
    assert launches.order == ["a", "c", "b"]


def test_an_entry_that_cannot_start_is_kept_as_failed_and_the_next_one_starts(jobs: TrainingJobs, launches: Launches):
    queue = TrainingQueue()
    runner = QueueRunner(queue, jobs, launches)
    launches.broken.add("a")
    a, b = queue.add(request("a")), queue.add(request("b"))
    runner.tick()
    assert launches.order == ["b"]
    assert [(e.request.run, e.state, e.error) for e in queue.entries()] == [("a", "failed", "a cannot start")]
    outcome = runner.outcome(a.id)
    assert outcome.state == "failed_to_start" and outcome.error == "a cannot start" and outcome.done and outcome.run_dir is None
    with pytest.raises(RuntimeError):
        runner.run_now(a.id)  # still broken: the error reaches the caller and the entry stays
    assert queue.entries()[0].state == "failed"
    launches.broken.clear()
    assert runner.run_now(a.id).run_dir.name == "a"
    assert queue.entries() == [] and runner.outcome(a.id).state == "running" and runner.outcome(b.id).state == "running"


def test_concurrent_ticks_never_exceed_max_parallel(jobs: TrainingJobs, launches: Launches):
    queue = TrainingQueue()
    runner = QueueRunner(queue, jobs, launches)
    queue.set_max_parallel(2)
    for i in range(6):
        queue.add(request(f"run-{i}"))
    threads = [threading.Thread(target=runner.tick) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert launches.order == ["run-0", "run-1"] and len(runner.running()) == 2


# --------------------------------------------------------------------------- the ticker


def test_ticker_keeps_ticking_after_an_error_and_stops_for_good():
    calls: list[int] = []

    def tick() -> None:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("first tick fails")

    ticker = Ticker(tick, interval=0.01)
    ticker.start()
    ticker.start()  # no second thread
    wait_for(lambda: len(calls) >= 3)
    ticker.stop()
    count = len(calls)
    time.sleep(0.05)
    assert len(calls) == count
    ticker.stop()  # harmless
