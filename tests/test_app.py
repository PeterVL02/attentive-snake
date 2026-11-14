"""The web app's API (FastAPI TestClient against temporary runs and rewards directories)."""

import shutil
import sys
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from attentive_snake.app import create_app
from attentive_snake.app.config_diff import differing_fields
from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.jobs import JobConflict, TrainingJobs
from attentive_snake.app.paths import PROJECT_ROOT, AppPaths, default_paths
from attentive_snake.app.watch_sessions import UnknownSession, WatchSessions
from attentive_snake.config import ExperimentConfig, save_experiment
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_env, make_policy
from attentive_snake.run_dir import config_path, list_checkpoints
from attentive_snake.run_metadata import RunEvent, append_event
from attentive_snake.training import save_checkpoint
from attentive_snake.training.checkpoint import checkpoint_path

KINDS = ["transformer", "cnn", "hybrid", "temporal_transformer", "frame_attention"]
JOB_TIMEOUT = 180.0


def tiny_tables(kind: str = "transformer", total_env_steps: int = 64, **train) -> dict:
    """A config that trains in a few seconds on the CPU."""
    history = 2 if kind in ("temporal_transformer", "frame_attention") else 1
    return {
        "game": {"width": 4, "height": 4, "initial_length": 2, "max_steps_without_food": 32},
        "reward": {},
        "model": {"kind": kind, "d_model": 16, "num_heads": 2, "num_layers": 1, "mlp_ratio": 2, "conv_channels": 8, "conv_layers": 1},
        "train": {
            "total_env_steps": total_env_steps,
            "rollout_length": 8,
            "num_epochs": 1,
            "num_minibatches": 2,
            "log_interval": 1,
            "checkpoint_interval": 2,
            "device": "cpu",
            **train,
        },
        "env": {"num_envs": 2, "history_length": history, "history_actions": False},
    }


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    rewards = tmp_path / "rewards"
    rewards.mkdir()
    shutil.copy(PROJECT_ROOT / "configs" / "rewards" / "default.toml", rewards / "default.toml")
    real = default_paths()
    return AppPaths(
        runs_dir=tmp_path / "runs",
        rewards_dir=rewards,
        default_config=real.default_config,
        train_script=real.train_script,
        cheat_sheet=real.cheat_sheet,
        working_dir=real.working_dir,
    )


@pytest.fixture
def client(paths: AppPaths):
    with TestClient(create_app(paths)) as test_client:  # exiting runs the shutdown (stops jobs)
        yield test_client


def make_run(paths: AppPaths, name: str, kind: str = "transformer", **train) -> Path:
    """A run directory with a config and one untrained checkpoint, built in-process (no subprocess)."""
    config: ExperimentConfig = experiment_from_tables(tiny_tables(kind, **train))
    run_dir = paths.runs_dir / name
    save_experiment(config, config_path(run_dir))
    policy = make_policy(config, make_env(config))
    save_checkpoint(checkpoint_path(run_dir, 16), policy, torch.optim.Adam(policy.parameters()), 16)
    return run_dir


def wait_until_done(client: TestClient, name: str) -> dict:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{name}").json()
        if detail["status"]["state"] != "running":
            return detail
        time.sleep(0.5)
    raise AssertionError(f"job {name} still running after {JOB_TIMEOUT}s")


# --------------------------------------------------------------------------- meta


def test_schema_defaults_and_page(client: TestClient):
    schema = client.get("/api/schema").json()
    assert set(schema["sections"]) == {"game", "reward", "model", "train", "env"}
    assert set(schema["model_kinds"]) == set(KINDS)
    defaults = client.get("/api/defaults").json()
    assert defaults["model"]["kind"] == "transformer"
    assert defaults["train"]["target_kl"] is None  # nullable fields are present as null
    assert client.get("/").status_code == 200
    script = client.get("/static/js/main.js")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
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

    assert not paths.runs_dir.exists() or not any(paths.runs_dir.iterdir())  # nothing was created

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
    listed = {run["name"]: run for run in client.get("/api/runs").json()}
    assert listed["tiny"]["env_steps"] == 64 and listed["tiny"]["kind"] == "transformer"

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
    assert len(list_checkpoints(paths.runs_dir / "noted")) > 1  # keep_checkpoints defaults to true

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
    assert "train.learning_rate" not in locked and "env.num_envs" not in locked


def test_fork_through_the_api(client: TestClient, paths: AppPaths):
    parent = make_run(paths, "parent")
    append_event(parent, RunEvent("start", 0, "parent run"))
    parent_files = {path.name: path.stat().st_mtime_ns for path in parent.iterdir()}
    tables = tiny_tables(total_env_steps=32, learning_rate=1e-3)
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
    assert not (paths.runs_dir / "child").exists()

    (paths.runs_dir / "empty").mkdir()
    save_experiment(experiment_from_tables(tiny_tables()), config_path(paths.runs_dir / "empty"))
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
    assert (paths.rewards_dir / "mine.toml").is_file()
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
    frame = body["frame"]
    assert frame["width"] == 4 and len(frame["heatmap"]) == 4

    steps = client.post(f"/api/watch/{session}/step", json={"count": 5}).json()["frames"]
    assert len(steps) == 5 and all("snake" in f for f in steps)
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
    empty = paths.runs_dir / "empty"
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
    differing = {(d["section"], d["field"]): d["values"] for d in body["differences"]}
    assert differing[("model", "kind")] == {"a": "transformer", "b": "cnn"}
    assert differing[("train", "learning_rate")] == {"a": 2.5e-4, "b": 1e-3}
    assert ("game", "width") not in differing
    assert client.post("/api/compare", json={"runs": ["a", "../b"]}).status_code == 422
    assert client.post("/api/compare", json={"runs": ["a", "missing"]}).status_code == 404
    assert client.post("/api/compare", json={"runs": []}).status_code == 422


def test_differing_fields():
    base = experiment_to_json(ExperimentConfig())
    other = experiment_to_json(experiment_from_tables({"game": {"width": 6}, "train": {"target_kl": 0.02}}))
    diff = differing_fields({"x": base, "y": other})
    assert [(d["section"], d["field"]) for d in diff] == [("game", "width"), ("train", "target_kl")]
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
    assert not (paths.rewards_dir.parent / "evil.toml").exists()


def test_watch_rejects_checkpoint_outside_the_run(client: TestClient, paths: AppPaths):
    make_run(paths, "safe")
    other = make_run(paths, "other")
    for checkpoint in ["../other/checkpoint_0000000016.pt", str(other / "checkpoint_0000000016.pt"), "config.toml"]:
        assert client.post("/api/watch", json={"run": "safe", "checkpoint": checkpoint}).status_code == 404
