"""The web app's API (FastAPI TestClient against temporary runs and rewards directories)."""

import base64
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import attentive_snake.games as games_package
from attentive_snake.app import create_app
from attentive_snake.app.background import BackgroundJobs, JobBusy
from attentive_snake.app.config_diff import differing_fields
from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.jobs import JobConflict, TrainingJobs
from attentive_snake.app.paths import PROJECT_ROOT, AppPaths, default_paths
from attentive_snake.app.train_command import resume_command
from attentive_snake.app.watch_sessions import UnknownSession, WatchSessions
from attentive_snake.config import ExperimentConfig, save_experiment
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.layout import default_config_path, rewards_dir
from attentive_snake.config.reward_schemes import save_scheme
from attentive_snake.evaluation.generic_statistics import GENERIC_METRICS
from attentive_snake.factory import make_env, make_policy
from attentive_snake.models.registry import KIND_INFO
from attentive_snake.games.registry import get_game, reload_games
from attentive_snake.run_dir import config_path, list_checkpoints
from attentive_snake.run_metadata import RunEvent, append_event
from attentive_snake.training import save_checkpoint
from attentive_snake.training.checkpoint import checkpoint_path

KINDS = ["transformer", "cnn", "resnet", "hybrid", "temporal_transformer", "frame_attention", "lstm"]
JOB_TIMEOUT = 180.0
DUMMY_GAMES = Path(__file__).resolve().parent / "dummy_games"
BEACON = {"name": "beacon", "length": 4, "max_steps": 8}  # dummy game: image view, no reward table


TINY_ALGORITHMS = {
    "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
    "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8, "target_update_interval": 16},
}


def tiny_tables(kind: str = "transformer", total_env_steps: int = 64, algorithm: str = "ppo", **train) -> dict:
    """A config that trains in a few seconds on the CPU."""
    history = 2 if kind in ("temporal_transformer", "frame_attention", "lstm") else 1
    return {
        "game": {"width": 4, "height": 4, "initial_length": 2, "max_steps_without_food": 32},
        "reward": {},
        "model": {"kind": kind, "d_model": 16, "num_heads": 2, "num_layers": 1, "mlp_ratio": 2, "conv_channels": 8, "conv_layers": 1},
        "train": {
            "total_env_steps": total_env_steps,
            "log_interval": 1,
            "checkpoint_interval": 32,  # every second update of 16 env steps
            "device": "cpu",
            "algorithm": algorithm,
            **train,
        },
        algorithm: dict(TINY_ALGORITHMS[algorithm]),
        "env": {"num_envs": 2, "history_length": history, "history_actions": False},
    }


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    configs = tmp_path / "configs"
    (configs / "snake" / "rewards").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "configs" / "snake" / "default.toml", configs / "snake" / "default.toml")
    shutil.copy(PROJECT_ROOT / "configs" / "snake" / "rewards" / "default.toml", configs / "snake" / "rewards" / "default.toml")
    shutil.copytree(PROJECT_ROOT / "configs" / "snake" / "algorithms", configs / "snake" / "algorithms")
    real = default_paths()
    return AppPaths(
        runs_dir=tmp_path / "runs",
        configs_dir=configs,
        train_script=real.train_script,
        cheat_sheet=real.cheat_sheet,
        working_dir=real.working_dir,
    )


@pytest.fixture
def client(paths: AppPaths):
    with TestClient(create_app(paths)) as test_client:  # exiting runs the shutdown (stops jobs)
        yield test_client


@pytest.fixture
def dummy_games(monkeypatch: pytest.MonkeyPatch):
    """Register the games of tests/dummy_games (walk, beacon) for one test; restored afterwards."""
    monkeypatch.setattr(games_package, "__path__", [*games_package.__path__, str(DUMMY_GAMES)])
    reload_games()
    yield
    monkeypatch.undo()
    reload_games()
    for name in [name for name in sys.modules if name.startswith(f"{games_package.__name__}.")]:
        if Path(getattr(sys.modules[name], "__file__", "") or "").is_relative_to(DUMMY_GAMES):
            del sys.modules[name]


@pytest.fixture
def multi_paths(paths: AppPaths, dummy_games) -> AppPaths:
    """``paths`` plus default configs and reward schemes of the dummy games; training subprocesses see them too."""
    for game in ("walk", "beacon"):
        spec = get_game(game)
        save_experiment(experiment_from_tables({"game": {"name": game}}), default_config_path(paths.configs_dir, game))
        save_scheme(rewards_dir(paths.configs_dir, game), "default", spec.reward_type())
    return AppPaths(
        runs_dir=paths.runs_dir,
        configs_dir=paths.configs_dir,
        train_script=DUMMY_GAMES / "train_with_dummy_games.py",
        cheat_sheet=paths.cheat_sheet,
        working_dir=paths.working_dir,
    )


@pytest.fixture
def multi_client(multi_paths: AppPaths):
    with TestClient(create_app(multi_paths)) as test_client:
        yield test_client


def make_run(paths: AppPaths, name: str, kind: str = "transformer", game: dict | None = None, **train) -> Path:
    """A run directory with a config and one untrained checkpoint, built in-process (no subprocess).

    ``game`` is the ``[game]`` table (default: a tiny Snake board).
    """
    tables = tiny_tables(kind, **train)
    if game is not None:
        tables["game"] = game
    config: ExperimentConfig = experiment_from_tables(tables)
    run_dir = paths.runs_dir / config.game_name / name
    save_experiment(config, config_path(run_dir))
    policy = make_policy(config, make_env(config))
    save_checkpoint(checkpoint_path(run_dir, 16), policy, torch.optim.Adam(policy.parameters()), 16)
    return run_dir


def wait_until_done(client: TestClient, name: str, game: str = "snake") -> dict:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{name}", params={"game": game}).json()
        if detail["status"]["state"] != "running":
            return detail
        time.sleep(0.5)
    raise AssertionError(f"job {name} still running after {JOB_TIMEOUT}s")


# --------------------------------------------------------------------------- meta


def test_games(client: TestClient):
    games = {game["name"]: game for game in client.get("/api/games").json()}
    metrics = games["snake"].pop("evaluation_metrics")
    assert games["snake"] == {
        "name": "snake", "title": "Snake", "description": games["snake"]["description"],
        "observation_kind": "grid", "view_kind": "board", "has_reward_table": True,
    }
    keys = [metric["key"] for metric in metrics]
    assert keys[:2] == ["win_rate", "mean_steps_to_win"] and "mean_score" in keys and "games" in keys
    assert set(metrics[0]) == {"key", "label", "description", "higher_is_better", "format", "unit"}
    for url in ("/api/schema", "/api/defaults", "/api/fork-rules", "/api/rewards", "/api/runs", "/api/runs/x"):
        response = client.get(url, params={"game": "../etc"})
        assert response.status_code == 404 and "unknown game" in response.json()["detail"], url
    assert client.post("/api/compare", json={"game": "chess", "runs": ["a"]}).status_code == 404
    assert client.post("/api/watch", json={"game": "chess", "run": "a"}).status_code == 404
    unknown = tiny_tables()
    unknown["game"]["name"] = "chess"
    assert client.post("/api/runs", json={"name": "x", "config": unknown}).status_code == 422


def test_schema_defaults_and_page(client: TestClient):
    assert client.get("/api/schema", params={"game": "snake"}).json() == client.get("/api/schema").json()
    schema = client.get("/api/schema").json()
    assert schema["game"] == "snake"
    assert set(schema["sections"]) == {"game", "reward", "model", "train", "env", "evaluation", *schema["algorithms"]}
    assert schema["algorithms"]["ppo"]["section"] == "ppo" and "ppo" in schema["sections"]
    assert set(schema["model_kinds"]) >= set(KINDS)  # more kinds may exist (e.g. for vector observations)
    defaults = client.get("/api/defaults").json()
    assert defaults["model"]["kind"] == "transformer" and defaults["game"]["name"] == "snake"
    assert defaults["ppo"]["target_kl"] is None  # nullable fields are present as null
    # The forms are built from the schema: bootstrap_truncated shows up with its help text (off for Snake).
    bootstrap = next(row for row in schema["sections"]["train"] if row["name"] == "bootstrap_truncated")
    assert bootstrap["type"] == "bool" and "cut short by a limit" in bootstrap["description"]
    assert defaults["train"]["bootstrap_truncated"] is False
    assert client.get("/").status_code == 200
    script = client.get("/static/js/main.js")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert client.get("/static/js/fork_form.js").status_code == 200
    assert "# " in client.get("/api/cheatsheet").text


# --------------------------------------------------------------------------- starting runs


def test_start_rejects_invalid_requests(client: TestClient, paths: AppPaths):
    bad_name = client.post("/api/runs", json={"name": "../escape", "config": tiny_tables()})
    assert bad_name.status_code == 422 and "invalid name" in bad_name.json()["detail"]

    tables = tiny_tables()
    tables["train"]["learning_rate"] = -1.0
    bad_value = client.post("/api/runs", json={"name": "bad-value", "config": tables})
    assert bad_value.status_code == 422 and "learning_rate" in bad_value.json()["detail"]

    tables = tiny_tables()
    tables["model"]["no_such_field"] = 1
    assert client.post("/api/runs", json={"name": "bad-key", "config": tables}).status_code == 422

    tables = tiny_tables()
    tables["game"]["width"] = "wide"
    assert client.post("/api/runs", json={"name": "bad-type", "config": tables}).status_code == 422

    assert not (paths.runs_dir / "snake").exists() or not any((paths.runs_dir / "snake").iterdir())  # nothing was created

    make_run(paths, "taken")
    existing = client.post("/api/runs", json={"name": "taken", "config": tiny_tables()})
    assert existing.status_code == 409


def test_train_then_resume_through_the_api(client: TestClient):
    started = client.post("/api/runs", json={"name": "tiny", "config": tiny_tables(total_env_steps=64)})
    assert started.status_code == 201, started.text
    detail = wait_until_done(client, "tiny")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["checkpoints"][-1]["steps"] == 64
    assert detail["metrics"] and detail["metrics"][-1]["env_steps"] == 64
    assert "mean_score" in detail["metrics"][-1] or "policy_loss" in detail["metrics"][-1]
    assert detail["log"]
    assert detail["config"]["train"]["total_env_steps"] == 64
    assert detail["game"] == "snake" and detail["config"]["game"]["name"] == "snake"
    listed = {run["name"]: run for run in client.get("/api/runs").json()}
    assert listed["tiny"]["env_steps"] == 64 and listed["tiny"]["kind"] == "transformer"
    assert listed["tiny"]["rules"]["width"] == 4 and "name" not in listed["tiny"]["rules"]
    assert client.get("/api/runs", params={"game": "snake"}).json()[0]["name"] == "tiny"

    resumed = client.post("/api/runs/tiny/resume", json={"total_env_steps": 128})
    assert resumed.status_code == 200, resumed.text
    detail = wait_until_done(client, "tiny")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["checkpoints"][-1]["steps"] == 128
    assert detail["total_env_steps"] == 128
    assert client.post("/api/runs/tiny/resume", json={"total_env_steps": 0}).status_code == 422


def test_stop_a_long_job_and_refuse_a_second_one(client: TestClient):
    started = client.post("/api/runs", json={"name": "long", "config": tiny_tables(total_env_steps=10_000_000)})
    assert started.status_code == 201
    assert client.post("/api/runs/long/resume", json={}).status_code == 409  # already training
    stopped = client.post("/api/runs/long/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"]["state"] == "stopped"
    assert client.post("/api/runs/long/stop").status_code == 409  # nothing running any more
    assert client.get("/api/runs/long").json()["status"]["state"] == "stopped"


def test_descriptions_and_keep_checkpoints_through_the_api(client: TestClient, paths: AppPaths):
    tables = tiny_tables(total_env_steps=64)
    started = client.post("/api/runs", json={"name": "noted", "config": tables, "description": "  -first try  "})
    assert started.status_code == 201, started.text
    assert wait_until_done(client, "noted")["status"]["state"] == "finished"
    assert len(list_checkpoints(paths.runs_dir / "snake" / "noted")) > 1  # keep_checkpoints defaults to true

    resumed = client.post(
        "/api/runs/noted/resume", json={"total_env_steps": 96, "keep_checkpoints": False, "description": "longer"}
    )
    assert resumed.status_code == 200, resumed.text
    detail = wait_until_done(client, "noted")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert [c["name"] for c in detail["checkpoints"]] == ["checkpoint_0000000096.pt"]
    assert detail["config"]["train"]["keep_checkpoints"] is False
    assert [(e["action"], e["description"], e["env_steps_at_action"]) for e in detail["history"]] == [
        ("start", "-first try", 0),
        ("resume", "longer", 64),
    ]
    listed = {run["name"]: run for run in client.get("/api/runs").json()}
    assert listed["noted"]["description"] == "longer" and listed["noted"]["parent"] is None
    compared = client.post("/api/compare", json={"runs": ["noted"]}).json()
    assert compared["runs"][0]["description"] == "longer"
    too_long = client.post("/api/runs/noted/resume", json={"description": "x" * 2001})
    assert too_long.status_code == 422


def test_fork_rules(client: TestClient):
    locked = client.get("/api/fork-rules").json()["locked"]
    assert "model.kind" in locked and "game.width" in locked and "env.history_length" in locked
    assert "game.name" in locked and client.get("/api/fork-rules", params={"game": "snake"}).json()["locked"] == locked
    assert "train.learning_rate" not in locked and "env.num_envs" not in locked


def test_fork_through_the_api(client: TestClient, paths: AppPaths):
    parent = make_run(paths, "parent")
    append_event(parent, RunEvent("start", 0, "parent run"))
    parent_files = {path.name: path.stat().st_mtime_ns for path in parent.iterdir()}
    tables = tiny_tables(total_env_steps=32, learning_rate=1e-3, bootstrap_truncated=True)
    tables["reward"] = {"food": 2.0}
    forked = client.post(
        "/api/runs/parent/fork",
        json={"name": "parent-v2", "checkpoint": "checkpoint_0000000016.pt", "config": tables, "description": "lr up"},
    )
    assert forked.status_code == 201, forked.text
    assert forked.json()["status"]["mode"] == "fork"
    detail = wait_until_done(client, "parent-v2")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["checkpoints"][-1]["steps"] == 32  # its own counter
    assert detail["config"]["train"]["learning_rate"] == 1e-3 and detail["config"]["reward"]["food"] == 2.0
    assert detail["config"]["train"]["bootstrap_truncated"] is True  # a training setting a fork may change
    assert detail["parent"] == {"run": "parent", "checkpoint": "checkpoint_0000000016.pt", "env_steps": 16}
    assert detail["description"] == "lr up"
    assert [e["action"] for e in detail["history"]] == ["fork"]
    assert {path.name: path.stat().st_mtime_ns for path in parent.iterdir()} == parent_files  # parent untouched

    latest = client.post("/api/runs/parent/fork", json={"name": "parent-v3", "config": tiny_tables()})
    assert latest.status_code == 201, latest.text
    assert wait_until_done(client, "parent-v3")["parent"]["checkpoint"] == "checkpoint_0000000016.pt"


def test_fork_validation(client: TestClient, paths: AppPaths):
    make_run(paths, "parent")
    make_run(paths, "taken")

    def fork(parent: str = "parent", **body) -> dict:
        return client.post(f"/api/runs/{parent}/fork", json={"name": "child", "config": tiny_tables()} | body)

    assert fork("missing").status_code == 404
    assert fork(name="../escape").status_code == 422
    assert fork(name="taken").status_code == 409
    assert fork(name="parent").status_code == 409
    assert fork(checkpoint="checkpoint_0000000099.pt").status_code == 404
    assert fork(checkpoint="../taken/checkpoint_0000000016.pt").status_code == 404
    for section, field, value in [("model", "d_model", 32), ("game", "width", 5), ("env", "history_length", 2)]:
        tables = tiny_tables()
        tables[section][field] = value
        response = fork(config=tables)
        assert response.status_code == 422 and f"{section}.{field}" in response.json()["detail"]
    tables = tiny_tables()
    tables["train"]["learning_rate"] = -1.0
    assert fork(config=tables).status_code == 422
    assert not (paths.runs_dir / "snake" / "child").exists()

    save_experiment(experiment_from_tables(tiny_tables()), config_path(paths.runs_dir / "snake" / "empty"))
    assert fork("empty").status_code == 409  # no checkpoint to continue from


def test_unknown_runs(client: TestClient):
    assert client.get("/api/runs/missing").status_code == 404
    assert client.post("/api/runs/missing/resume", json={}).status_code == 404
    assert client.post("/api/runs/missing/stop").status_code == 404


# --------------------------------------------------------------------------- architecture


@pytest.mark.parametrize("kind", KINDS)
def test_architecture_for_each_kind(client: TestClient, kind: str):
    response = client.post("/api/architecture", json=tiny_tables(kind))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == kind
    assert body["parameters"] > 0
    assert body["text"] and body["mermaid"].lstrip().startswith(("flowchart", "graph", "%%"))
    assert (body["algorithm"], body["head"]) == ("ppo", "actor_critic") and "value_head" in body["text"]
    q = client.post("/api/architecture", json=tiny_tables(kind, algorithm="dqn")).json()
    assert (q["algorithm"], q["head"]) == ("dqn", "q_values") and "q_head" in q["text"] and "value_head" not in q["text"]
    assert q["parameters"] < body["parameters"]  # one linear head instead of two: the trunk is the same


def test_architecture_validation_and_runs(client: TestClient, paths: AppPaths):
    tables = tiny_tables()
    tables["model"]["num_heads"] = 3  # d_model 16 is not divisible by 3
    assert client.post("/api/architecture", json=tables).status_code == 422
    make_run(paths, "arch")
    assert client.get("/api/runs/arch/architecture").json()["parameters"] > 0


# --------------------------------------------------------------------------- reward schemes


def test_reward_scheme_crud_and_preview(client: TestClient, paths: AppPaths):
    assert client.get("/api/rewards").json()["schemes"] == ["default"]
    scheme = {"food": 2.0, "death": -5, "step": 0.0}
    saved = client.put("/api/rewards/mine", json=scheme)
    assert saved.status_code == 200, saved.text
    assert (paths.configs_dir / "snake" / "rewards" / "mine.toml").is_file()
    assert client.get("/api/rewards", params={"game": "snake"}).json()["schemes"] == ["default", "mine"]
    assert client.get("/api/rewards").json()["schemes"] == ["default", "mine"]
    loaded = client.get("/api/rewards/mine").json()
    assert loaded["food"] == 2.0 and loaded["death"] == -5.0 and loaded["win"] == 1.0

    preview = client.post("/api/rewards/preview", json={"reward": scheme, "example_score": 3}).json()
    situations = {row["situation"]: row for row in preview}
    assert situations["Hit a wall"]["reward"] == -5.0
    assert any("score 3" in s for s in situations)

    assert client.put("/api/rewards/mine", json={"food": "lots"}).status_code == 422
    assert client.put("/api/rewards/mine", json={"unknown": 1}).status_code == 422
    assert client.post("/api/rewards/preview", json={"reward": {"food": "x"}}).status_code == 422

    assert client.delete("/api/rewards/default").status_code == 403
    assert client.delete("/api/rewards/mine").status_code == 204
    assert client.get("/api/rewards/mine").status_code == 404
    assert client.delete("/api/rewards/mine").status_code == 404


# --------------------------------------------------------------------------- watching


def test_watch_session_lifecycle(client: TestClient, paths: AppPaths):
    make_run(paths, "watched")
    created = client.post("/api/watch", json={"run": "watched", "checkpoint": "checkpoint_0000000016.pt"})
    assert created.status_code == 201, created.text
    body = created.json()
    session = body["id"]
    assert body["info"]["kind"] == "transformer" and body["info"]["attention_view"] == "grid"
    assert body["info"]["game"] == "snake" and body["info"]["view_kind"] == "board"
    frame = body["frame"]
    assert frame["width"] == 4 and len(frame["heatmap"]) == 4 and frame["view_kind"] == "board"
    assert client.post("/api/watch", json={"game": "snake", "run": "watched"}).status_code == 201

    steps = client.post(f"/api/watch/{session}/step", json={"count": 5}).json()["frames"]
    assert len(steps) == 5 and all("snake" in f for f in steps)
    last_only = client.post(f"/api/watch/{session}/step", json={"count": 3, "frames": "last"}).json()["frames"]
    assert [set(f) for f in last_only[:-1]] == [{"score", "steps", "event", "done", "episode"}] * 2
    assert "snake" in last_only[-1] and "heatmap" in last_only[-1]
    assert client.post(f"/api/watch/{session}/step", json={"frames": "some"}).status_code == 422
    assert client.post(f"/api/watch/{session}/step", json={"count": 0}).status_code == 422
    assert client.post(f"/api/watch/{session}/step", json={"count": 1000}).status_code == 422

    options = client.post(f"/api/watch/{session}/options", json={"attention": False, "sample": True})
    assert options.status_code == 200 and options.json()["frame"]["heatmap"] is None
    assert client.post(f"/api/watch/{session}/options", json={"layer": 7}).status_code == 422
    assert client.post(f"/api/watch/{session}/options", json={"frame": -5}).status_code == 422
    assert client.post(f"/api/watch/{session}/reset").json()["frame"]["steps"] == 0

    assert client.delete(f"/api/watch/{session}").status_code == 204
    assert client.post(f"/api/watch/{session}/step", json={}).status_code == 404
    assert client.delete(f"/api/watch/{session}").status_code == 404


def test_watch_model_without_attention_and_bad_requests(client: TestClient, paths: AppPaths):
    make_run(paths, "conv", kind="cnn")
    body = client.post("/api/watch", json={"run": "conv"}).json()
    assert body["info"]["attention_view"] is None and body["frame"]["heatmap"] is None
    assert client.post("/api/watch", json={"run": "nope"}).status_code == 404
    assert client.post("/api/watch", json={"run": "conv", "checkpoint": "checkpoint_9.pt"}).status_code == 404
    assert client.post("/api/watch", json={"run": "conv", "layer": 5}).status_code == 201  # no attention: any layer
    make_run(paths, "attn")
    assert client.post("/api/watch", json={"run": "attn", "layer": 5}).status_code == 422
    empty = paths.runs_dir / "snake" / "empty"
    save_experiment(experiment_from_tables(tiny_tables()), config_path(empty))
    assert client.post("/api/watch", json={"run": "empty"}).status_code == 404  # no checkpoint yet


def test_watch_sessions_evict_least_recently_used():
    sessions = WatchSessions(capacity=2)
    first, second = sessions.add("a"), sessions.add("b")
    with sessions.use(first):
        pass  # first is now the most recently used
    sessions.add("c")
    assert len(sessions) == 2
    with pytest.raises(UnknownSession):
        with sessions.use(second):
            pass
    with sessions.use(first) as session:
        assert session == "a"


# --------------------------------------------------------------------------- jobs


def test_training_jobs_conflict_stop_and_shutdown(tmp_path: Path):
    jobs = TrainingJobs(tmp_path, stop_timeout=5)
    sleeper = [sys.executable, "-c", "import time; print('hi', flush=True); time.sleep(60)"]
    log = tmp_path / "run" / "train.log"
    assert jobs.launch("run", sleeper, log, "new")["state"] == "running"
    with pytest.raises(JobConflict):
        jobs.launch("run", sleeper, log, "resume")
    assert jobs.stop("run")["state"] == "stopped"
    with pytest.raises(LookupError):
        jobs.stop("run")
    jobs.launch("other", sleeper, tmp_path / "other.log", "new")
    jobs.shutdown()
    assert jobs.status("other")["state"] == "stopped"
    failing = TrainingJobs(tmp_path)
    failing.launch("bad", [sys.executable, "-c", "raise SystemExit(3)"], log, "new")
    deadline = time.monotonic() + 30
    while failing.is_running("bad") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert failing.status("bad") | {"pid": 0, "started_at": 0} == {
        "state": "failed", "exit_code": 3, "mode": "new", "pid": 0, "started_at": 0,
    }
    assert "=== " in log.read_text()


# --------------------------------------------------------------------------- comparing


def test_compare_runs(client: TestClient, paths: AppPaths):
    make_run(paths, "a")
    make_run(paths, "b", kind="cnn", learning_rate=1e-3)
    response = client.post("/api/compare", json={"runs": ["a", "b", "a"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [run["name"] for run in body["runs"]] == ["a", "b"]
    assert all(run["parameters"] > 0 and run["metrics"] == [] for run in body["runs"])
    assert body["game"] == "snake"
    differing = {(d["section"], d["field"]): d["values"] for d in body["differences"]}
    assert differing[("model", "kind")] == {"a": "transformer", "b": "cnn"}
    assert differing[("train", "learning_rate")] == {"a": 2.5e-4, "b": 1e-3}
    assert ("game", "width") not in differing
    assert client.post("/api/compare", json={"runs": ["a", "../b"]}).status_code == 422
    assert client.post("/api/compare", json={"runs": ["a", "missing"]}).status_code == 404
    assert client.post("/api/compare", json={"runs": []}).status_code == 422


def test_differing_fields():
    base = experiment_to_json(ExperimentConfig())
    other = experiment_to_json(experiment_from_tables({"game": {"width": 6}, "ppo": {"target_kl": 0.02}}))
    diff = differing_fields({"x": base, "y": other})
    assert [(d["section"], d["field"]) for d in diff] == [("game", "width"), ("ppo", "target_kl")]
    assert diff[1]["values"] == {"x": None, "y": 0.02}
    assert differing_fields({"x": base}) == []
    assert differing_fields({"x": {"s": {"v": 1}}, "y": {"s": {"v": 1.0}}}) == []  # int vs float is equal
    assert differing_fields({"x": {"s": {"v": 1}}, "y": {"s": {"v": True}}})[0]["field"] == "v"
    assert differing_fields({"x": {"s": {"v": 1}}, "y": {"s": {}}})[0]["values"] == {"x": 1, "y": None}


# --------------------------------------------------------------------------- path traversal


@pytest.mark.parametrize(
    "method, url, body",
    [
        ("get", "/api/runs/..", None),
        ("get", "/api/runs/.hidden", None),
        ("get", "/api/runs/..%2F..%2Fetc", None),
        ("post", "/api/runs/..%2Fx/resume", {}),
        ("post", "/api/runs/..%2Fx/fork", {"name": "y", "config": {}}),
        ("get", "/api/runs/..%2Fx/architecture", None),
        ("get", "/api/rewards/..%2F..%2Fpyproject", None),
        ("put", "/api/rewards/..%2Fevil", {"food": 1.0}),
        ("delete", "/api/rewards/..%2Fdefault", None),
        ("get", "/static/..%2F..%2Fpaths.py", None),
    ],
)
def test_path_traversal_is_rejected(client: TestClient, paths: AppPaths, method: str, url: str, body):
    response = client.request(method.upper(), url, json=body)
    assert response.status_code in (400, 404, 405, 422), (url, response.status_code, response.text)
    assert not (paths.configs_dir / "snake" / "evil.toml").exists()


def test_watch_rejects_checkpoint_outside_the_run(client: TestClient, paths: AppPaths):
    make_run(paths, "safe")
    other = make_run(paths, "other")
    for checkpoint in ["../other/checkpoint_0000000016.pt", str(other / "checkpoint_0000000016.pt"), "config.toml"]:
        assert client.post("/api/watch", json={"run": "safe", "checkpoint": checkpoint}).status_code == 404


# --------------------------------------------------------------------------- several games


def test_games_lists_every_registered_game(multi_client: TestClient):
    games = {game["name"]: game for game in multi_client.get("/api/games").json()}
    assert {"snake", "walk", "beacon"} <= set(games)
    beacon_metrics = [metric["key"] for metric in games["beacon"].pop("evaluation_metrics")]
    assert beacon_metrics == [definition.key for definition in GENERIC_METRICS]  # no metrics of its own
    walk_metrics = [metric["key"] for metric in games["walk"]["evaluation_metrics"]]
    assert walk_metrics[0] == "reach_rate"  # a dropped-in game's own metrics come first
    assert games["beacon"] == {
        "name": "beacon", "title": "Beacon", "description": "Drift right until you reach the beacon.",
        "observation_kind": "grid", "view_kind": "image", "has_reward_table": False,
    }
    assert games["walk"]["view_kind"] == "board" and games["walk"]["has_reward_table"] is False


def test_schema_defaults_and_rewards_per_game(multi_client: TestClient, multi_paths: AppPaths):
    schema = multi_client.get("/api/schema", params={"game": "beacon"}).json()
    assert schema["game"] == "beacon"
    assert [row["name"] for row in schema["sections"]["game"]] == ["length", "max_steps"]
    assert [row["name"] for row in schema["sections"]["reward"]] == ["beacon", "step"]
    assert set(schema["model_kinds"]) >= set(KINDS)
    snake_schema = multi_client.get("/api/schema").json()
    assert "width" in [row["name"] for row in snake_schema["sections"]["game"]]

    defaults = multi_client.get("/api/defaults", params={"game": "beacon"}).json()
    assert defaults["game"] == {"name": "beacon", "length": 5, "max_steps": 12}
    assert "game.length" in multi_client.get("/api/fork-rules", params={"game": "beacon"}).json()["locked"]

    assert multi_client.get("/api/rewards", params={"game": "beacon"}).json()["schemes"] == ["default"]
    assert multi_client.get("/api/rewards/default", params={"game": "beacon"}).json() == {"beacon": 1.0, "step": -0.01}
    saved = multi_client.put("/api/rewards/eager", params={"game": "beacon"}, json={"beacon": 5.0})
    assert saved.status_code == 200 and saved.json()["game"] == "beacon"
    assert (multi_paths.configs_dir / "beacon" / "rewards" / "eager.toml").is_file()
    assert multi_client.get("/api/rewards").json()["schemes"] == ["default"]  # snake's schemes are separate
    assert multi_client.put("/api/rewards/bad", params={"game": "beacon"}, json={"food": 1.0}).status_code == 422
    preview = multi_client.post("/api/rewards/preview", params={"game": "beacon"}, json={"reward": {}})
    assert preview.status_code == 404 and "no reward preview" in preview.json()["detail"]
    assert multi_client.post("/api/rewards/preview", json={"reward": {}}).status_code == 200


def test_train_a_run_of_another_game_through_the_api(multi_client: TestClient, multi_paths: AppPaths):
    tables = tiny_tables(total_env_steps=32)
    tables["game"] = BEACON
    started = multi_client.post("/api/runs", json={"name": "tiny", "config": tables})
    assert started.status_code == 201, started.text
    assert started.json()["game"] == "beacon"
    detail = wait_until_done(multi_client, "tiny", game="beacon")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["game"] == "beacon" and detail["rules"] == {"length": 4, "max_steps": 8}
    assert detail["checkpoints"][-1]["steps"] == 32
    assert (multi_paths.runs_dir / "beacon" / "tiny" / "config.toml").is_file()
    assert not (multi_paths.runs_dir / "snake" / "tiny").exists()
    assert [run["name"] for run in multi_client.get("/api/runs", params={"game": "beacon"}).json()] == ["tiny"]
    assert multi_client.get("/api/runs").json() == []
    assert multi_client.get("/api/runs/tiny").status_code == 404  # not a snake run
    assert multi_client.get("/api/runs/tiny/architecture", params={"game": "beacon"}).json()["game"] == "beacon"


def test_runs_and_compare_never_mix_games(multi_client: TestClient, multi_paths: AppPaths):
    make_run(multi_paths, "same")
    make_run(multi_paths, "same", kind="cnn", game=BEACON)
    make_run(multi_paths, "only-beacon", game=BEACON)
    snake = multi_client.get("/api/runs/same").json()
    beacon = multi_client.get("/api/runs/same", params={"game": "beacon"}).json()
    assert (snake["game"], snake["kind"]) == ("snake", "transformer")
    assert (beacon["game"], beacon["kind"]) == ("beacon", "cnn")
    assert [run["name"] for run in multi_client.get("/api/runs").json()] == ["same"]
    assert {run["name"] for run in multi_client.get("/api/runs", params={"game": "beacon"}).json()} == {"same", "only-beacon"}

    compared = multi_client.post("/api/compare", json={"game": "beacon", "runs": ["same", "only-beacon"]}).json()
    assert compared["game"] == "beacon"
    assert {run["config"]["game"]["name"] for run in compared["runs"]} == {"beacon"}
    assert ("model", "kind") in {(d["section"], d["field"]) for d in compared["differences"]}
    assert multi_client.post("/api/compare", json={"runs": ["same", "only-beacon"]}).status_code == 404
    assert multi_client.post("/api/compare", json={"runs": ["same"]}).json()["runs"][0]["config"]["game"]["name"] == "snake"


def test_watch_a_game_with_an_image_view(multi_client: TestClient, multi_paths: AppPaths):
    make_run(multi_paths, "seen", game=BEACON)
    assert multi_client.post("/api/watch", json={"run": "seen"}).status_code == 404  # a beacon run, not a snake run
    created = multi_client.post("/api/watch", json={"game": "beacon", "run": "seen", "seed": 0})
    assert created.status_code == 201, created.text
    body = created.json()
    assert (body["info"]["game"], body["info"]["view_kind"]) == ("beacon", "image")
    frame = body["frame"]
    assert frame["view_kind"] == "image" and (frame["width"], frame["height"]) == (16, 4)
    assert base64.b64decode(frame["png"]).startswith(b"\x89PNG") and "snake" not in frame
    steps = multi_client.post(f"/api/watch/{body['id']}/step", json={"count": 4, "frames": "last"}).json()["frames"]
    assert "png" in steps[-1] and all("png" not in f for f in steps[:-1])


UNKNOWN_GAME_REQUESTS = [
    ("get", "/api/schema", None),
    ("get", "/api/defaults", None),
    ("get", "/api/fork-rules", None),
    ("get", "/api/rewards", None),
    ("get", "/api/rewards/default", None),
    ("put", "/api/rewards/default", {"food": 1.0}),
    ("delete", "/api/rewards/mine", None),
    ("post", "/api/rewards/preview", {"reward": {}}),
    ("get", "/api/runs", None),
    ("get", "/api/runs/x", None),
    ("get", "/api/runs/x/architecture", None),
    ("post", "/api/runs/x/resume", {}),
    ("post", "/api/runs/x/fork", {"name": "y", "config": {}}),
    ("post", "/api/runs/x/stop", None),
]


@pytest.mark.parametrize("game", ["chess", "../snake", "snake/..", "..", "SNAKE", ""])
@pytest.mark.parametrize("method, url, body", UNKNOWN_GAME_REQUESTS)
def test_unknown_or_malicious_games_are_rejected(client: TestClient, paths: AppPaths, game: str, method: str, url: str, body):
    make_run(paths, "x")
    response = client.request(method.upper(), url, params={"game": game}, json=body)
    assert response.status_code == 404 and "unknown game" in response.json()["detail"], (url, response.text)
    assert sorted(path.name for path in paths.configs_dir.rglob("*.toml")) == ["default.toml", "default.toml", "dqn.toml"]
    assert sorted(path.name for path in paths.runs_dir.iterdir()) == ["snake"]


@pytest.mark.parametrize("game", ["chess", "../snake", ".."])
def test_unknown_or_malicious_games_in_bodies_are_rejected(client: TestClient, paths: AppPaths, game: str):
    make_run(paths, "x")
    assert client.post("/api/compare", json={"game": game, "runs": ["x"]}).status_code == 404
    assert client.post("/api/watch", json={"game": game, "run": "x"}).status_code == 404
    tables = tiny_tables()
    tables["game"]["name"] = game
    assert client.post("/api/runs", json={"name": "new", "config": tables}).status_code == 422
    assert sorted(path.name for path in paths.runs_dir.iterdir()) == ["snake"]


def test_watch_lunar_lander_with_attention_over_its_numbers(client: TestClient, paths: AppPaths):
    """A real Gymnasium game: the session resets before its first view (Gymnasium cannot render before a reset)."""
    make_run(paths, "lander", game={"name": "lunar_lander"})
    created = client.post("/api/watch", json={"game": "lunar_lander", "run": "lander", "seed": 0})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["info"]["element_names"][:2] == ["x", "y"] and len(body["info"]["element_names"]) == 8
    assert (body["frame"]["width"], body["frame"]["height"]) == (600, 400) and body["frame"]["heatmap"] is None
    steps = client.post(f"/api/watch/{body['id']}/step", json={"count": 3, "frames": "last"}).json()["frames"]
    assert "png" in steps[-1] and len(steps[-1]["element_weights"]) == 8


def test_lunar_lander_hidden_velocities_in_the_schema_and_a_watched_lstm(client: TestClient, paths: AppPaths):
    """The rule is a form row, every kind reads the vector observation, and a kind without attention shows no weights."""
    schema = client.get("/api/schema", params={"game": "lunar_lander"}).json()
    rows = {row["name"]: row for row in schema["sections"]["game"]}
    assert rows["hide_velocities"]["type"] == "bool" and rows["hide_velocities"]["default"] is False
    for kind in ("resnet", "lstm"):
        info = schema["model_kinds"][kind]
        assert "vector" in info["observation_kinds"] and not info["recommended"] and "kernel_size" not in info["fields"]
        assert info["description"] == KIND_INFO[kind].inputs["vector"].description
    assert "num_layers" in schema["model_kinds"]["lstm"]["fields"] and schema["model_kinds"]["lstm"]["uses_history"]
    assert "game.hide_velocities" in client.get("/api/fork-rules", params={"game": "lunar_lander"}).json()["locked"]

    make_run(paths, "blind", kind="lstm", game={"name": "lunar_lander", "hide_velocities": True})
    body = client.post("/api/watch", json={"game": "lunar_lander", "run": "blind", "seed": 0}).json()
    assert body["info"]["attention_view"] is None and body["info"]["element_names"][2] == "vx"
    frame = client.post(f"/api/watch/{body['id']}/step", json={"count": 2, "frames": "last"}).json()["frames"][-1]
    assert frame["element_weights"] is None and frame["frame_weights"] is None and frame["heatmap"] is None
    detail = client.get("/api/runs/blind", params={"game": "lunar_lander"}).json()
    assert detail["config"]["game"]["hide_velocities"] is True


# --------------------------------------------------------------------------- evaluation


def wait_for_evaluation(client: TestClient, name: str, game: str = "snake") -> dict:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{name}/evaluate", params={"game": game}).json()
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"evaluation of {name} still running after {JOB_TIMEOUT}s")


def test_training_evaluations_are_listed_and_reproduced_through_the_api(client: TestClient):
    tables = tiny_tables(total_env_steps=64, keep_checkpoints=False)  # 16 env steps per update
    tables["evaluation"] = {"interval": 32, "games": 4}
    assert client.post("/api/runs", json={"name": "graded", "config": tables}).status_code == 201
    detail = wait_until_done(client, "graded")
    assert detail["status"]["state"] == "finished", detail["log"]
    recorded = detail["evaluations"]
    assert [(e["env_steps"], e["source"], e["checkpoint_available"]) for e in recorded] == [
        (32, "training", False),  # pruned by keep_checkpoints = false: can no longer be reproduced
        (64, "training", True),
    ]
    assert recorded[1]["settings"]["games"] == 4 and "win_rate" in recorded[1]["statistics"]
    assert "evaluation steps=32" in detail["log"]

    gone = client.post("/api/runs/graded/evaluate", json={"checkpoint": recorded[0]["checkpoint"]})
    assert gone.status_code == 404
    started = client.post("/api/runs/graded/evaluate", json={"checkpoint": recorded[1]["checkpoint"], "games": 4, "seed": 1000, "save": True})
    assert started.status_code == 202, started.text
    status = wait_for_evaluation(client, "graded")
    assert status["state"] == "finished", status
    result = status["result"]
    assert result["reproduces"] == {"timestamp": recorded[1]["timestamp"], "source": "training", "identical": True, "same_device": True}
    assert result["evaluation"]["statistics"] == recorded[1]["statistics"] and result["saved"] is True
    assert status["progress"] == {"done": 4, "total": 4} and status["request"]["checkpoint"] == recorded[1]["checkpoint"]
    detail = client.get("/api/runs/graded").json()
    assert [e["source"] for e in detail["evaluations"]] == ["training", "training", "manual"]
    compared = client.post("/api/compare", json={"runs": ["graded"]}).json()
    assert len(compared["runs"][0]["evaluations"]) == 3

    resumed = client.post("/api/runs/graded/resume", json={"total_env_steps": 80, "evaluation_interval": 16, "evaluation_games": 2})
    assert resumed.status_code == 200, resumed.text
    detail = wait_until_done(client, "graded")
    assert detail["config"]["evaluation"]["interval"] == 16 and detail["config"]["evaluation"]["games"] == 2
    assert [(e["env_steps"], e["settings"]["games"]) for e in detail["evaluations"][3:]] == [(80, 2)]
    assert client.post("/api/runs/graded/resume", json={"evaluation_interval": -1}).status_code == 422


def test_evaluation_requests_are_validated_and_one_runs_at_a_time(client: TestClient, paths: AppPaths):
    make_run(paths, "solo")
    assert client.get("/api/runs/solo/evaluate").json() == {"state": "idle"}
    assert client.get("/api/runs/solo").json()["evaluations"] == []
    assert client.post("/api/runs/missing/evaluate", json={}).status_code == 404
    assert client.post("/api/runs/solo/evaluate", params={"game": "chess"}, json={}).status_code == 404
    assert client.get("/api/runs/solo/evaluate", params={"game": "../x"}).status_code == 404
    assert client.post("/api/runs/solo/evaluate", json={"checkpoint": "checkpoint_9.pt"}).status_code == 404
    assert client.post("/api/runs/solo/evaluate", json={"games": 0}).status_code == 422
    assert client.post("/api/runs/..%2Fx/evaluate", json={}).status_code in (404, 422)

    first = client.post("/api/runs/solo/evaluate", json={"games": 10_000})
    assert first.status_code == 202 and first.json()["state"] == "running"
    second = client.post("/api/runs/solo/evaluate", json={"games": 1})
    assert second.status_code == 409 and "already running" in second.json()["detail"]
    status = wait_for_evaluation(client, "solo")
    assert status["state"] == "finished" and status["result"]["evaluation"]["statistics"]["games"] == 10_000
    assert status["result"]["reproduces"] is None and status["result"]["saved"] is False
    assert client.get("/api/runs/solo").json()["evaluations"] == []  # not saved

    empty = paths.runs_dir / "snake" / "empty"
    save_experiment(experiment_from_tables(tiny_tables()), config_path(empty))
    assert client.post("/api/runs/empty/evaluate", json={}).status_code == 409  # no checkpoint yet


def test_background_jobs_report_failures_and_refuse_a_second_job():
    jobs = BackgroundJobs()
    release = threading.Event()

    def blocked(progress):
        progress(1, 2)
        release.wait(10)
        return {"ok": True}

    assert jobs.start("a", blocked, {"n": 1}, 2)["state"] == "running"
    with pytest.raises(JobBusy):
        jobs.start("a", blocked, {}, 1)
    assert jobs.start("b", lambda progress: 1 / 0, {}, 1)  # other keys are independent
    release.set()
    deadline = time.monotonic() + 10
    while jobs.status("a")["state"] == "running" or jobs.status("b")["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert jobs.status("a")["result"] == {"ok": True} and jobs.status("a")["progress"]["done"] == 1
    assert jobs.status("b")["state"] == "failed" and "ZeroDivisionError" in jobs.status("b")["error"]
    assert jobs.status("c") is None


def test_resume_command_passes_evaluation_changes():
    command = resume_command(Path("train.py"), Path("runs/snake/x"), evaluation={"interval": 0, "games": 5})
    assert command[-4:] == ["--eval-interval", "0", "--eval-games", "5"]


def test_the_forms_config_with_evaluation_trains_and_a_config_without_it_still_resumes(client: TestClient, paths: AppPaths):
    """The New-training form sends /api/defaults (which has [evaluation]); the train.py subprocess must load it.
    Runs from before [evaluation] existed have no such section and must still load (evaluation off)."""
    form = client.get("/api/defaults").json()
    assert "evaluation" in form
    small = tiny_tables(total_env_steps=32)
    for section in ("game", "model", "train", "ppo", "env"):
        form[section].update(small[section])
    form["train"]["checkpoint_dir"] = None
    form["evaluation"].update({"interval": 16, "games": 2})
    assert client.post("/api/runs", json={"name": "form", "config": form}).status_code == 201
    detail = wait_until_done(client, "form")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert "[evaluation]" in config_path(paths.runs_dir / "snake" / "form").read_text()
    assert [e["env_steps"] for e in detail["evaluations"]] == [16, 32]

    config_file = config_path(paths.runs_dir / "snake" / "form")
    text = config_file.read_text()
    config_file.write_text(text[: text.index("[evaluation]")])  # as written before the section existed
    assert client.get("/api/runs/form").json()["config"]["evaluation"]["interval"] == 0
    assert client.post("/api/runs/form/resume", json={"total_env_steps": 48}).status_code == 200
    detail = wait_until_done(client, "form")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert len(detail["evaluations"]) == 2  # evaluation off for the resumed part


# --------------------------------------------------------------------------- algorithms


def wait_for_evaluation(client: TestClient, name: str) -> dict:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{name}/evaluate").json()
        if status["state"] not in ("running", "idle"):
            return status
        time.sleep(0.2)
    raise AssertionError(f"evaluation of {name} still running after {JOB_TIMEOUT}s")


def test_algorithm_defaults_and_schema(client: TestClient):
    schema = client.get("/api/schema").json()
    assert set(schema["algorithms"]) == {"ppo", "dqn"}
    assert schema["algorithms"]["dqn"]["head"] == "q_values" and "loss" in [m["key"] for m in schema["algorithms"]["dqn"]["metrics"]]
    algorithm_row = next(row for row in schema["sections"]["train"] if row["name"] == "algorithm")
    assert algorithm_row["choices"] == ["ppo", "dqn"]
    ppo = client.get("/api/defaults").json()
    dqn = client.get("/api/defaults", params={"algorithm": "dqn"}).json()
    assert ppo["train"]["algorithm"] == "ppo" and "ppo" in ppo and "dqn" not in ppo
    assert dqn["train"]["algorithm"] == "dqn" and "dqn" in dqn and "ppo" not in dqn
    assert dqn["dqn"]["buffer_size"] == 100_000 and dqn["env"]["num_envs"] == 16  # configs/snake/algorithms/dqn.toml
    assert dqn["model"] == ppo["model"] and dqn["game"] == ppo["game"]
    assert client.get("/api/defaults", params={"algorithm": "a2c"}).status_code == 404


def test_prioritized_replay_settings_are_in_the_schema_and_a_prioritized_run_logs_its_metrics(client: TestClient):
    schema = client.get("/api/schema").json()
    fields = {row["name"]: row for row in schema["sections"]["dqn"]}
    for name in ("prioritized", "priority_alpha", "priority_beta_start", "priority_epsilon"):
        assert fields[name]["description"], name  # the new-run form and the fork dialog are built from these
    assert [m["key"] for m in schema["algorithms"]["dqn"]["metrics"] if m["setting"] == "prioritized"] == ["priority_beta", "priority_mean", "priority_max"]
    assert client.get("/api/defaults", params={"algorithm": "dqn"}).json()["dqn"]["prioritized"] is False
    tables = tiny_tables("cnn", algorithm="dqn")
    tables["dqn"] |= {"prioritized": True, "priority_beta_start": 0.5}
    assert client.post("/api/runs", json={"name": "per", "config": tables}).status_code == 201
    detail = wait_until_done(client, "per")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["config"]["dqn"]["prioritized"] is True
    assert {"priority_beta", "priority_mean", "priority_max"} <= set(detail["metrics"][-1]) and detail["metrics"][-1]["priority_beta"] == 1.0
    tables["dqn"]["priority_alpha"] = 0.3  # changeable on fork
    forked = client.post("/api/runs/per/fork", json={"name": "per-v2", "config": tables})
    assert forked.status_code == 201, forked.text
    assert wait_until_done(client, "per-v2")["config"]["dqn"]["priority_alpha"] == 0.3


def test_a_dqn_run_through_the_api_trains_watches_evaluates_forks_and_compares(client: TestClient, paths: AppPaths):
    started = client.post("/api/runs", json={"name": "dqn", "config": tiny_tables("cnn", algorithm="dqn")})
    assert started.status_code == 201, started.text
    detail = wait_until_done(client, "dqn")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["algorithm"] == "dqn" and detail["config"]["train"]["algorithm"] == "dqn" and "dqn" in detail["config"]
    assert "loss" in detail["metrics"][-1] and detail["metrics"][-1]["env_steps"] == 64
    assert next(run for run in client.get("/api/runs").json() if run["name"] == "dqn")["algorithm"] == "dqn"

    watch = client.post("/api/watch", json={"run": "dqn"}).json()
    assert watch["info"]["algorithm"] == "dqn" and watch["info"]["action_names"] == ["up", "right", "down", "left"]
    assert watch["frame"]["action_values"]["kind"] == "q_values" and len(watch["frame"]["action_values"]["values"]) == 4
    assert client.post(f"/api/watch/{watch['id']}/step", json={"count": 2}).status_code == 200

    assert client.post("/api/runs/dqn/evaluate", json={"games": 3}).status_code == 202
    evaluated = wait_for_evaluation(client, "dqn")
    assert evaluated["state"] == "finished", evaluated
    assert evaluated["result"]["evaluation"]["statistics"]["games"] == 3

    changed = tiny_tables("cnn", total_env_steps=32, algorithm="dqn")
    changed["dqn"]["epsilon_start"] = 0.2
    assert client.post("/api/runs/dqn/fork", json={"name": "dqn-v2", "config": changed}).status_code == 201
    other = tiny_tables("cnn", algorithm="ppo")
    refused = client.post("/api/runs/dqn/fork", json={"name": "as-ppo", "config": other})
    assert refused.status_code == 422 and "train.algorithm" in refused.json()["detail"]
    child = wait_until_done(client, "dqn-v2")
    assert child["status"]["state"] == "finished", child["log"]
    assert child["config"]["dqn"]["epsilon_start"] == 0.2 and child["parent"]["run"] == "dqn"

    make_run(paths, "ppo", kind="cnn")
    compared = client.post("/api/compare", json={"runs": ["dqn", "ppo"]}).json()
    assert [run["algorithm"] for run in compared["runs"]] == ["dqn", "ppo"]
    differing = {(d["section"], d["field"]) for d in compared["differences"]}
    assert ("train", "algorithm") in differing and ("dqn", "buffer_size") in differing and ("ppo", "rollout_length") in differing
    architecture = client.get("/api/runs/dqn/architecture").json()
    assert "q_head" in json.dumps(architecture) and "policy_head" not in json.dumps(architecture)


def test_a_run_whose_config_cannot_be_read_is_listed_and_shown_not_a_server_error(client: TestClient, paths: AppPaths):
    run_dir = make_run(paths, "old")
    config_path(run_dir).write_text('[train]\nrollout_length = 128\n')  # a [train] key from before [ppo] existed
    listed = next(run for run in client.get("/api/runs").json() if run["name"] == "old")
    assert listed["kind"] is None and "rollout_length" in listed["config_error"]
    detail = client.get("/api/runs/old")
    assert detail.status_code == 200 and detail.json()["config"] is None and "rollout_length" in detail.json()["config_error"]
    make_run(paths, "new")
    compared = client.post("/api/compare", json={"runs": ["old", "new"]})
    assert compared.status_code == 200 and compared.json()["runs"][0]["config"] is None
    assert client.get("/api/runs/old/architecture").status_code == 422
    assert client.post("/api/watch", json={"run": "old"}).status_code == 422
    assert client.post("/api/runs/old/evaluate", json={}).status_code == 422
