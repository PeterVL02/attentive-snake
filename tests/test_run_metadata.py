"""The run history file (run.json)."""

import datetime
import json

import pytest

from attentive_snake.run_dir import metadata_path
from attentive_snake.run_metadata import (
    Lineage,
    RunEvent,
    append_event,
    clean_description,
    event_to_json,
    latest_description,
    lineage,
    read_events,
)


def test_missing_file_is_an_empty_history(tmp_path) -> None:
    assert read_events(tmp_path) == []
    assert latest_description([]) is None and lineage([]) is None


def test_append_and_read_round_trip(tmp_path) -> None:
    run_dir = tmp_path / "run"
    fork = RunEvent("fork", 0, "lower lr", Lineage("parent", "checkpoint_0000000512.pt", 512))
    resume = RunEvent("resume", 1024)
    append_event(run_dir, fork)
    append_event(run_dir, resume)
    assert read_events(run_dir) == [fork, resume]
    assert latest_description([fork, resume]) == "lower lr"
    assert lineage([fork, resume]) == Lineage("parent", "checkpoint_0000000512.pt", 512)
    assert [p.name for p in run_dir.iterdir()] == ["run.json"]  # no temporary file left behind

    data = json.loads(metadata_path(run_dir).read_text())
    assert data["events"][0]["parent"] == {"run": "parent", "checkpoint": "checkpoint_0000000512.pt", "env_steps": 512}
    assert "parent" not in data["events"][1] and data["events"][1]["description"] is None
    timestamp = datetime.datetime.fromisoformat(data["events"][0]["timestamp"])
    assert timestamp.utcoffset() == datetime.timedelta(0)


def test_latest_description_skips_events_without_one() -> None:
    events = [RunEvent("start", 0, "first"), RunEvent("resume", 10, "second"), RunEvent("resume", 20)]
    assert latest_description(events) == "second"


def test_event_validation() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        RunEvent("restart", 0)
    with pytest.raises(ValueError, match="parent"):
        RunEvent("fork", 0)
    with pytest.raises(ValueError, match="parent"):
        RunEvent("start", 0, parent=Lineage("p", "c.pt", 1))
    assert "parent" not in event_to_json(RunEvent("start", 0))


def test_malformed_file_is_a_value_error(tmp_path) -> None:
    metadata_path(tmp_path).write_text("{not json")
    with pytest.raises(ValueError, match="run.json"):
        read_events(tmp_path)
    metadata_path(tmp_path).write_text('{"events": [{"action": "start"}]}')
    with pytest.raises(ValueError):
        read_events(tmp_path)


@pytest.mark.parametrize("text, expected", [(None, None), ("", None), ("   \n", None), ("  hi there \n", "hi there")])
def test_clean_description(text, expected) -> None:
    assert clean_description(text) == expected
