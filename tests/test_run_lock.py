"""The lock that allows one training process per run directory."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from attentive_snake import run_lock
from attentive_snake.run_dir import lock_path
from attentive_snake.run_lock import RunLocked, hold_run_lock, lock_holder

pytest.importorskip("fcntl")

HOLDER = """
import fcntl, os, sys, time
descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)
fcntl.flock(descriptor, fcntl.LOCK_EX)
os.write(descriptor, str(os.getpid()).encode())
print("locked", flush=True)
time.sleep(120)
"""


def test_a_run_that_never_trained_has_no_holder(tmp_path: Path):
    assert lock_holder(tmp_path / "missing") is None
    assert lock_holder(tmp_path) is None


def test_the_holder_is_known_until_the_block_ends_and_the_file_stays(tmp_path: Path):
    run_dir = tmp_path / "new" / "run"
    with hold_run_lock(run_dir):
        assert lock_holder(run_dir) == os.getpid()
    assert lock_holder(run_dir) is None
    assert lock_path(run_dir).is_file()


def test_a_second_holder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(run_lock, "_ACQUIRE_ATTEMPTS", 1)
    with hold_run_lock(tmp_path):
        with pytest.raises(RunLocked, match="already training"):
            with hold_run_lock(tmp_path):
                pass
    with hold_run_lock(tmp_path):  # free again
        pass


def test_the_lock_is_released_when_the_holder_is_killed(tmp_path: Path):
    process = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock_path(tmp_path))], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        assert lock_holder(tmp_path) == process.pid
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)
        assert lock_holder(tmp_path) is None
        with hold_run_lock(tmp_path):
            assert lock_holder(tmp_path) == os.getpid()
    finally:
        process.kill()
        process.wait()
        process.stdout.close()
