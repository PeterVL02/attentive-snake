import os

import pytest

from attentive_snake.run_dir import (
    checkpoint_steps,
    config_path,
    game_runs_dir,
    games_with_runs,
    list_checkpoints,
    list_runs,
    metrics_path,
    validate_run_name,
)
from attentive_snake.run_migration import legacy_runs, migrate, plan_migration


def make_run(run_dir) -> None:
    config_path(run_dir).parent.mkdir(parents=True)
    config_path(run_dir).write_text("")
    (run_dir / "checkpoint_0000000010.pt").write_text("weights")


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


def test_runs_are_grouped_per_game(tmp_path) -> None:
    assert game_runs_dir(tmp_path, "snake") == tmp_path / "snake"
    assert games_with_runs(tmp_path / "missing") == []
    make_run(tmp_path / "snake" / "a")
    make_run(tmp_path / "legacy")  # a run directly in the root is not a game
    (tmp_path / "empty_game").mkdir()
    assert games_with_runs(tmp_path) == ["snake"]
    assert list_runs(game_runs_dir(tmp_path, "snake")) == ["a"]


def test_migration_moves_legacy_runs_only_and_is_idempotent(tmp_path) -> None:
    make_run(tmp_path / "old1")
    make_run(tmp_path / "old2")
    make_run(tmp_path / "snake" / "new")
    (tmp_path / "not_a_run").mkdir()
    assert legacy_runs(tmp_path) == [tmp_path / "old1", tmp_path / "old2"]
    planned = plan_migration(tmp_path)
    assert planned == [(tmp_path / "old1", tmp_path / "snake" / "old1"), (tmp_path / "old2", tmp_path / "snake" / "old2")]
    assert (tmp_path / "old1").exists()  # planning touches nothing
    assert migrate(tmp_path) == planned
    assert sorted(list_runs(tmp_path / "snake")) == ["new", "old1", "old2"]
    assert (tmp_path / "snake" / "old1" / "checkpoint_0000000010.pt").read_text() == "weights"
    assert (tmp_path / "not_a_run").is_dir()
    assert migrate(tmp_path) == []  # nothing left to do


def test_migration_refuses_to_overwrite_and_moves_nothing(tmp_path) -> None:
    make_run(tmp_path / "a")
    make_run(tmp_path / "b")
    make_run(tmp_path / "snake" / "b")
    with pytest.raises(FileExistsError, match="snake/b"):
        migrate(tmp_path)
    assert legacy_runs(tmp_path) == [tmp_path / "a", tmp_path / "b"]


def test_migration_refuses_a_legacy_run_named_like_the_game(tmp_path) -> None:
    make_run(tmp_path / "snake")
    with pytest.raises(FileExistsError, match="itself a run"):
        migrate(tmp_path)
