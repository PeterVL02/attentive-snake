"""Training jobs: one ``scripts/train.py`` subprocess per run, launched, watched and stopped from the server.

Only jobs started by this ``TrainingJobs`` are known; a run trained from a terminal
shows up as having no job. Output goes to the log file given at launch.
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

JobState = Literal["running", "finished", "failed", "stopped"]


class JobConflict(RuntimeError):
    """A job for this run is already running."""


@dataclass
class _Job:
    process: subprocess.Popen
    log: IO[bytes]
    mode: str
    started_at: float = field(default_factory=time.time)
    stop_requested: bool = False

    def state(self) -> JobState:
        code = self.process.poll()
        if code is None:
            return "running"
        if self.stop_requested:
            return "stopped"
        return "finished" if code == 0 else "failed"

    def describe(self) -> dict:
        return {
            "state": self.state(),
            "exit_code": self.process.returncode,
            "mode": self.mode,
            "pid": self.process.pid,
            "started_at": self.started_at,
        }


def _signal(process: subprocess.Popen, signum: int) -> None:
    """Signal the job's whole process group (it is started in its own session)."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        elif signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


class TrainingJobs:
    """At most one running training subprocess per run name."""

    def __init__(self, working_dir: Path, stop_timeout: float = 10.0) -> None:
        self._working_dir = working_dir
        self._stop_timeout = stop_timeout
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def launch(self, run: str, command: list[str], log_path: Path, mode: str) -> dict:
        """Start ``command`` for ``run`` with stdout and stderr appended to ``log_path``."""
        with self._lock:
            previous = self._jobs.get(run)
            if previous is not None and previous.state() == "running":
                raise JobConflict(f"run {run!r} is already training (pid {previous.process.pid}).")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab")
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {mode}: {' '.join(command)}\n".encode())
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=self._working_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            if previous is not None:
                previous.log.close()
            self._jobs[run] = _Job(process, log, mode)
            return self._jobs[run].describe()

    def status(self, run: str) -> dict | None:
        """The run's latest job (running or not), or ``None`` if none was started here."""
        with self._lock:
            job = self._jobs.get(run)
            return None if job is None else job.describe()

    def is_running(self, run: str) -> bool:
        status = self.status(run)
        return status is not None and status["state"] == "running"

    def stop(self, run: str) -> dict:
        """Terminate the run's job, killing it if it has not exited after the stop timeout."""
        with self._lock:
            job = self._jobs.get(run)
        if job is None or job.state() != "running":
            raise LookupError(f"run {run!r} has no running job.")
        self._stop(job)
        return job.describe()

    def shutdown(self) -> None:
        """Stop every running job (called when the server exits)."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.state() == "running":
                self._stop(job)
            job.log.close()

    def _stop(self, job: _Job) -> None:
        job.stop_requested = True
        _signal(job.process, signal.SIGTERM)
        try:
            job.process.wait(self._stop_timeout)
        except subprocess.TimeoutExpired:
            _signal(job.process, signal.SIGKILL)
            job.process.wait()
