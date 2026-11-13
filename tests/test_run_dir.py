import os

import pytest

from attentive_snake.run_dir import (
    checkpoint_steps,
    config_path,
    list_checkpoints,
    list_runs,
    metrics_path,
    validate_run_name,
)


def test_metrics_path(tmp_path) -> None:
    assert metrics_path(tmp_path) == tmp_path / "metrics.jsonl"


def test_list_runs_needs_config_and_sorts_newest_first(tmp_path) -> None:
    assert list_runs(tmp_path / "missing") == []
    for name, mtime in (("old", 1_000_000), ("new", 3_000_000), ("mid", 2_000_000)):
        config_path(tmp_path / name).parent.mkdir()
        config_path(tmp_path / name).write_text("")
        os.utime(tmp_path / name, (mtime, mtime))
    (tmp_path / "no_config").mkdir()
    (tmp_path / "file.txt").write_text("")
    assert list_runs(tmp_path) == ["new", "mid", "old"]


def test_list_checkpoints_sorted_by_steps_and_steps_parsed(tmp_path) -> None:
    assert list_checkpoints(tmp_path / "missing") == []
    for steps in (30, 5, 1000):
        (tmp_path / f"checkpoint_{steps:010d}.pt").write_text("")
    (tmp_path / "other.pt").write_text("")
    found = list_checkpoints(tmp_path)
    assert [checkpoint_steps(p) for p in found] == [5, 30, 1000]
    with pytest.raises(ValueError):
        checkpoint_steps(tmp_path / "other.pt")


@pytest.mark.parametrize("name", ["run1", "8x8_big-2", "a.b", "_x", "-x"])
def test_valid_run_names(name) -> None:
    assert validate_run_name(name) == name


@pytest.mark.parametrize("name", ["", ".hidden", "..", "../x", "a/b", "a\\b", "a b", "x\n", "é"])
def test_invalid_run_names(name) -> None:
    with pytest.raises(ValueError):
        validate_run_name(name)
