"""Calls a function at a fixed interval from a daemon thread: how the training queue notices that a job ended.

Polling, not a callback from ``TrainingJobs``: a subprocess does not announce its exit, so a
callback would need a waiting thread per job inside ``jobs.py``; one thread asking once a second
keeps ``jobs.py`` unaware of the queue.
"""

import logging
import threading
from collections.abc import Callable

_log = logging.getLogger(__name__)


class Ticker:
    def __init__(self, tick: Callable[[], None], interval: float = 1.0, name: str = "training queue") -> None:
        self._tick = tick
        self._name = name
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin calling ``tick`` every ``interval`` seconds (nothing happens if it already does)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop, and wait for a call in progress to return: afterwards ``tick`` is not running and will not be called."""
        thread, self._thread = self._thread, None
        self._stop.set()
        if thread is not None:
            thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._tick()
            except Exception:  # one bad tick must not end the queue
                _log.exception("%s tick failed", self._name)
