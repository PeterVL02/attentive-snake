"""Stopping training cleanly: at a whole update, with a checkpoint, and without ever leaving a truncated checkpoint."""

import signal
from pathlib import Path

import pytest
import torch
from torch import nn

from attentive_snake.training import NullLogger, TrainConfig, Trainer, latest_checkpoint, list_checkpoints, save_checkpoint
from attentive_snake.training.checkpoint import checkpoint_path
from attentive_snake.training.hooks import UpdateEnd
from attentive_snake.training.stop_signal import stop_on_signals

STEPS_PER_UPDATE = 4


class CountingLearner:
    """Learns nothing and counts its updates."""

    steps_per_iteration = STEPS_PER_UPDATE

    def __init__(self) -> None:
        self.policy = nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.updates = 0

    def iterate(self, env_steps: int) -> tuple:
        self.updates += 1
        return ()

    def report(self) -> dict[str, float]:
        return {}

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, state: dict) -> None:
        pass


def make_trainer(directory: Path, updates: int, hooks: tuple = ()) -> Trainer:
    config = TrainConfig(
        total_env_steps=updates * STEPS_PER_UPDATE, log_interval=STEPS_PER_UPDATE, checkpoint_interval=10_000, checkpoint_dir=directory, device="cpu"
    )
    return Trainer(CountingLearner(), config, torch.device("cpu"), NullLogger(), hooks)


def saved_steps(directory: Path) -> list[int]:
    return [int(path.stem.split("_")[1]) for path in list_checkpoints(directory)]


def test_a_stop_after_an_update_saves_it_and_skips_its_hooks(tmp_path: Path):
    hooked: list[int] = []
    trainer = make_trainer(tmp_path, updates=5, hooks=(lambda end: hooked.append(end.env_steps),))
    trainer.train(lambda: trainer.learner.updates >= 2)
    assert trainer.stopped_early and trainer.env_steps == 2 * STEPS_PER_UPDATE
    assert hooked == [STEPS_PER_UPDATE]
    assert saved_steps(tmp_path) == [2 * STEPS_PER_UPDATE]


def test_a_stop_during_the_hooks_ends_training_after_them(tmp_path: Path):
    requested: list[bool] = []
    trainer = make_trainer(tmp_path, updates=5, hooks=(lambda end: requested.append(True),))
    trainer.train(lambda: bool(requested))
    assert trainer.stopped_early and trainer.learner.updates == 1
    assert saved_steps(tmp_path) == [STEPS_PER_UPDATE]


def test_the_last_update_is_never_cut_short_so_the_final_evaluation_still_runs(tmp_path: Path):
    ends: list[UpdateEnd] = []
    trainer = make_trainer(tmp_path, updates=2, hooks=(ends.append,))
    trainer.train(lambda: trainer.learner.updates == 2)
    assert not trainer.stopped_early
    assert [end.is_last for end in ends] == [False, True]


def test_without_a_stop_nothing_changes(tmp_path: Path):
    trainer = make_trainer(tmp_path, updates=3)
    trainer.train()
    assert not trainer.stopped_early and trainer.learner.updates == 3
    assert saved_steps(tmp_path) == [3 * STEPS_PER_UPDATE]


def test_an_interrupted_save_leaves_the_previous_checkpoint_the_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    policy = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(policy.parameters())
    save_checkpoint(checkpoint_path(tmp_path, 4), policy, optimizer, 4)

    def interrupted(_state: dict, path: Path) -> None:
        Path(path).write_bytes(b"truncated")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(checkpoint_path(tmp_path, 8), policy, optimizer, 8)
    assert latest_checkpoint(tmp_path) == checkpoint_path(tmp_path, 4)


@pytest.mark.parametrize("signum, exit_code", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_the_first_signal_requests_a_stop_and_restores_the_handlers(signum: int, exit_code: int):
    before = signal.getsignal(signum)
    with stop_on_signals() as stop:
        assert not stop()
        signal.raise_signal(signum)
        assert stop() and stop.exit_code == exit_code
        assert signal.getsignal(signum) == before  # a second signal ends the process at once
    assert signal.getsignal(signum) == before


def test_the_handlers_are_restored_when_no_signal_came():
    before = signal.getsignal(signal.SIGTERM)
    with stop_on_signals():
        assert signal.getsignal(signal.SIGTERM) != before
    assert signal.getsignal(signal.SIGTERM) == before
