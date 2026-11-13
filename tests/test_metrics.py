import json

from attentive_snake.run_dir import metrics_path
from attentive_snake.training import ConsoleLogger, JsonlLogger, TeeLogger, read_metrics


def test_jsonl_logger_appends_flushes_and_creates_parents(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "metrics.jsonl"
    logger = JsonlLogger(path)
    logger.log(100, {"loss": 0.5, "env_steps": 100.0})
    assert json.loads(path.read_text().splitlines()[0]) == {"env_steps": 100, "loss": 0.5}  # visible before any close
    JsonlLogger(path).log(200, {"loss": 0.25})  # a resumed run appends
    assert read_metrics(path) == [{"env_steps": 100, "loss": 0.5}, {"env_steps": 200, "loss": 0.25}]


def test_read_metrics_tolerates_missing_file_and_truncated_last_line(tmp_path) -> None:
    assert read_metrics(tmp_path / "missing.jsonl") == []
    path = tmp_path / "m.jsonl"
    path.write_text('{"env_steps": 1, "x": 2.0}\n{"env_steps": 2, "x"')
    assert read_metrics(path) == [{"env_steps": 1, "x": 2.0}]


def test_tee_logger_forwards_to_every_logger(tmp_path, capsys) -> None:
    path = metrics_path(tmp_path)
    TeeLogger(ConsoleLogger(), JsonlLogger(path)).log(5, {"a": 1.0})
    assert "steps=5" in capsys.readouterr().out
    assert read_metrics(path) == [{"env_steps": 5, "a": 1.0}]
