"""Turning SIGTERM and SIGINT into a request to stop training after the current update.

The first signal only sets the request. The previous handlers are back in place from then on,
so a second signal (a second Ctrl-C) ends the process at once.
"""

import signal
from collections.abc import Iterator
from contextlib import contextmanager

_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class StopRequest:
    """Called, says whether a stop was requested (the ``should_stop`` of ``Trainer.train``)."""

    def __init__(self) -> None:
        self.signum: int | None = None

    def __call__(self) -> bool:
        return self.signum is not None

    @property
    def exit_code(self) -> int:
        """The shell's code for a process ended by this signal: 143 for SIGTERM, 130 for SIGINT."""
        assert self.signum is not None
        return 128 + self.signum


@contextmanager
def stop_on_signals() -> Iterator[StopRequest]:
    """Inside the block SIGTERM and SIGINT set the ``StopRequest``; the previous handlers are restored afterwards."""
    request = StopRequest()
    previous = {signum: signal.getsignal(signum) for signum in _SIGNALS}

    def restore() -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def handle(signum: int, _frame: object) -> None:
        request.signum = signum
        restore()

    for signum in _SIGNALS:
        signal.signal(signum, handle)
    try:
        yield request
    finally:
        restore()
