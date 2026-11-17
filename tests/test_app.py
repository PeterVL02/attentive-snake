"""The web app's API (FastAPI TestClient against temporary runs and rewards directories)."""

import base64
import json
import re
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
    history = 2 if KIND_INFO[kind].uses_history else 1
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
    assert set(metrics[0]) == {"key", "label", "description", "higher_is_better", "format", "unit", "measures_reward"}
    assert [metric["key"] for metric in metrics if metric["measures_reward"]] == ["mean_return"]
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
    assert client.get("/static/js/tabs/queue.js").status_code == 200
    assert client.get("/static/js/queue_panel.js").status_code == 200
    for script in ("tabs/reward_search.js", "reward_search_form.js", "reward_search_view.js"):
        assert client.get(f"/static/js/{script}").status_code == 200, script
    assert "# " in client.get("/api/cheatsheet").text


def test_every_named_js_import_is_exported_by_its_module():
    """No node here to load the frontend: at least every ``import { name } from "./x.js"`` must name an export of x.js."""
    js_dir = PROJECT_ROOT / "src" / "attentive_snake" / "app" / "static" / "js"
    imports = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"(\.{1,2}/[^"]+)"')
    exports = re.compile(r"export\s+(?:async\s+)?(?:function\*?|const|let|class)\s+(\w+)")
    missing = []
    for module in sorted(js_dir.rglob("*.js")):
        for names, target in imports.findall(module.read_text(encoding="utf-8")):
            path = (module.parent / target).resolve()
            exported = set(exports.findall(path.read_text(encoding="utf-8"))) if path.is_file() else set()
            for name in (n.split(" as ")[0].strip() for n in names.split(",")):
                if name and name not in exported:
                    missing.append(f"{module.relative_to(js_dir)}: {name} from {target}")
    assert missing == []


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


# --------------------------------------------------------------------------- parameter count of a form's config


def parameters_for(client: TestClient, tables: dict, game: str = "snake") -> int:
    response = client.post("/api/parameters", params={"game": game}, json={"config": tables})
    assert response.status_code == 200, response.text
    return response.json()["parameters"]


def test_parameter_count_equals_the_architecture_and_compare_of_a_run(client: TestClient, paths: AppPaths):
    tables = tiny_tables("cnn")
    count = parameters_for(client, tables)
    make_run(paths, "same", kind="cnn")
    assert count == client.get("/api/runs/same/architecture").json()["parameters"]
    assert count == client.post("/api/compare", json={"runs": ["same"]}).json()["runs"][0]["parameters"]
    assert count == client.post("/api/architecture", json=tables).json()["parameters"]


def test_parameter_count_follows_the_model_board_history_and_algorithm(client: TestClient):
    base = parameters_for(client, tiny_tables("cnn"))
    wider = tiny_tables("cnn")
    wider["model"]["conv_channels"] = 16
    assert parameters_for(client, wider) != base
    bigger_board = tiny_tables("cnn")
    bigger_board["game"].update(width=6, height=6)
    assert parameters_for(client, bigger_board) != base  # the cnn's readout depends on the board
    assert parameters_for(client, tiny_tables("cnn", algorithm="dqn")) < base  # one head instead of two

    for kind in ("temporal_transformer", "cnn"):  # a sequence of frames / extra input channels
        longer = tiny_tables(kind)
        longer["env"]["history_length"] = 3
        assert parameters_for(client, longer) != parameters_for(client, tiny_tables(kind))
    d_model = tiny_tables()
    d_model["model"]["d_model"] = 32
    assert parameters_for(client, d_model) != parameters_for(client, tiny_tables())


def test_parameter_count_for_other_games(multi_client: TestClient):
    beacon = tiny_tables("cnn")
    beacon["game"] = BEACON
    assert parameters_for(multi_client, beacon, game="beacon") > 0
    lander = {"model": {"kind": "mlp", "d_model": 16}, "env": {"num_envs": 2, "history_length": 1}}
    assert parameters_for(multi_client, lander, game="lunar_lander") > 0
    assert parameters_for(multi_client, tiny_tables("cnn")) != parameters_for(multi_client, beacon, game="beacon")


def test_parameter_count_errors_and_no_side_effects(client: TestClient, paths: AppPaths):
    tables = tiny_tables()
    tables["model"]["num_heads"] = 3  # d_model 16 is not divisible by 3
    bad = client.post("/api/parameters", params={"game": "snake"}, json={"config": tables})
    assert bad.status_code == 422 and bad.json()["detail"]
    unknown_field = tiny_tables()
    unknown_field["model"]["nonsense"] = 1
    assert client.post("/api/parameters", params={"game": "snake"}, json={"config": unknown_field}).status_code == 422
    assert client.post("/api/parameters", params={"game": "nope"}, json={"config": tiny_tables()}).status_code == 404
    assert client.post("/api/parameters", json={"config": tiny_tables()}).status_code == 422  # ?game= is required
    parameters_for(client, tiny_tables())
    assert not paths.runs_dir.exists() or list(paths.runs_dir.rglob("*")) == []


def training_warnings_for(client: TestClient, tables: dict, game: str = "snake"):
    return client.post("/api/training-warnings", params={"game": game}, json={"config": tables})


def test_training_warnings_route(client: TestClient, paths: AppPaths):
    tables = tiny_tables()  # default reward: step -0.005, death -1; gamma 0.99: below the limit
    assert training_warnings_for(client, tables).json() == {"warnings": []}
    tables["reward"] = {"step": -0.01}
    (text,) = training_warnings_for(client, tables).json()["warnings"]
    assert "step cost of -0.01" in text
    tables["train"]["gamma"] = 0.5
    assert training_warnings_for(client, tables).json() == {"warnings": []}
    assert not paths.runs_dir.exists()


def test_training_warnings_route_errors_and_other_games(client: TestClient, multi_client: TestClient):
    assert training_warnings_for(client, {"model": {"nonsense": 1}}).status_code == 422
    assert training_warnings_for(client, tiny_tables(), game="nope").status_code == 404
    lander = {"model": {"kind": "mlp", "d_model": 16}}
    assert training_warnings_for(multi_client, lander, game="lunar_lander").json() == {"warnings": []}
    assert training_warnings_for(multi_client, {}, game="beacon").json() == {"warnings": []}


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
    assert [run["kind"] for run in body["runs"]] == ["transformer", "cnn"]  # the Compare tab groups by it
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


def test_the_apple_rules_are_form_rows_and_a_run_with_them_trains_and_is_watched_with_every_apple(client: TestClient):
    schema = client.get("/api/schema").json()
    rows = {row["name"]: row for row in schema["sections"]["game"]}
    assert rows["food_distribution"]["choices"] == ["uniform", "center"] and rows["food_distribution"]["default"] == "uniform"
    assert rows["food_spread"]["type"] == "float" and rows["food_spread"]["default"] == 2.0 and rows["food_spread"]["choices"] is None
    assert rows["food_count"]["type"] == "int" and rows["food_count"]["default"] == 1
    assert all(rows[name]["description"] for name in ("food_distribution", "food_spread", "food_count"))
    assert client.get("/api/fork-rules").json()["locked"].keys().isdisjoint({"game.food_count", "game.food_distribution", "game.food_spread"})

    tables = tiny_tables()
    tables["game"].update({"food_count": 3, "food_distribution": "center", "food_spread": 1.5})
    assert client.post("/api/runs", json={"name": "apples", "config": tables}).status_code == 201
    detail = wait_until_done(client, "apples")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["config"]["game"]["food_count"] == 3 and detail["config"]["game"]["food_distribution"] == "center"
    assert client.get("/api/runs").json()[0]["rules"]["food_count"] == 3
    body = client.post("/api/watch", json={"run": "apples", "seed": 0}).json()
    assert len(body["frame"]["foods"]) == 3
    frames = client.post(f"/api/watch/{body['id']}/step", json={"count": 3}).json()["frames"]
    assert all(len(frame["foods"]) == 3 for frame in frames if "foods" in frame and not frame["done"])
    assert client.post("/api/runs", json={"name": "bad", "config": {**tables, "game": {**tables["game"], "food_count": 0}}}).status_code == 422


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
    assert [m["key"] for m in schema["algorithms"]["dqn"]["metrics"] if m["setting"] == "dqn.prioritized"] == ["priority_beta", "priority_mean", "priority_max"]
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


def test_n_step_dueling_and_loss_are_in_the_schema_and_reach_a_started_run(client: TestClient, paths: AppPaths):
    schema = client.get("/api/schema").json()
    fields = {row["name"]: row for row in schema["sections"]["dqn"]}
    assert (fields["n_step"]["type"], fields["n_step"]["default"], fields["n_step"]["choices"]) == ("int", 1, None)
    assert (fields["dueling"]["type"], fields["dueling"]["default"]) == ("bool", False)
    assert (fields["loss"]["default"], fields["loss"]["choices"]) == ("huber", ["huber", "mse"])  # a dropdown in the form
    assert all(len(fields[name]["description"]) > 80 for name in ("n_step", "dueling", "loss"))
    assert schema["algorithms"]["dqn"]["head"] == "q_values"  # with the default settings
    defaults = client.get("/api/defaults", params={"algorithm": "dqn"}).json()["dqn"]
    assert (defaults["n_step"], defaults["dueling"], defaults["loss"]) == (1, False, "huber")
    assert "dqn.dueling" in client.get("/api/fork-rules").json()["locked"]  # the fork dialog shows it locked
    assert not {"dqn.n_step", "dqn.loss"} & set(client.get("/api/fork-rules").json()["locked"])

    tables = tiny_tables("cnn", algorithm="dqn")
    plain_parameters = parameters_for(client, tables)
    tables["dqn"] |= {"n_step": 3, "dueling": True, "loss": "mse"}
    assert parameters_for(client, tables) > plain_parameters  # the two streams' hidden layers
    drawn = client.post("/api/architecture", json=tables).json()
    assert drawn["head"] == "dueling" and "dueling_advantage" in drawn["text"] and "q_head" not in drawn["text"]
    assert client.post("/api/runs", json={"name": "rainbowish", "config": tables}).status_code == 201
    detail = wait_until_done(client, "rainbowish")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["config"]["dqn"] | {"n_step": 3, "dueling": True, "loss": "mse"} == detail["config"]["dqn"]
    assert 'loss = "mse"' in config_path(paths.runs_dir / "snake" / "rainbowish").read_text()
    assert "loss" in detail["metrics"][-1]
    assert client.get("/api/runs/rainbowish/architecture").json()["head"] == "dueling"
    watch = client.post("/api/watch", json={"run": "rainbowish"}).json()
    assert watch["info"]["algorithm"] == "dqn" and len(watch["frame"]["action_values"]["values"]) == 4  # Q-values, as without it

    changed = tiny_tables("cnn", total_env_steps=32, algorithm="dqn")
    changed["dqn"] |= {"n_step": 1, "dueling": True, "loss": "huber"}  # n_step and loss may change on fork
    forked = client.post("/api/runs/rainbowish/fork", json={"name": "rainbowish-v2", "config": changed})
    assert forked.status_code == 201, forked.text
    assert wait_until_done(client, "rainbowish-v2")["status"]["state"] == "finished"
    changed["dqn"]["dueling"] = False
    refused = client.post("/api/runs/rainbowish/fork", json={"name": "undueled", "config": changed})
    assert refused.status_code == 422 and "dqn.dueling" in refused.text
    resumed = client.post("/api/runs/rainbowish/resume", json={"total_env_steps": 96})
    assert resumed.status_code == 200, resumed.text
    assert wait_until_done(client, "rainbowish")["status"]["state"] == "finished"  # the dueling weights load again

    jepa = tiny_tables("jepa", algorithm="dqn")
    jepa["dqn"]["dueling"] = True
    assert "advantage_decoder" in client.post("/api/architecture", json=jepa).json()["text"]
    jepa["model"] |= {"value_training": "actual", "prediction_coef": 1.0}
    refused = client.post("/api/parameters", params={"game": "snake"}, json={"config": jepa})  # what the New-training form asks while it is filled in
    assert refused.status_code == 422 and "dueling" in refused.text


def test_jepa_is_offered_drawn_trained_watched_and_forked_through_the_api(client: TestClient):
    schema = client.get("/api/schema").json()
    kind = schema["model_kinds"]["jepa"]
    assert {"prediction_coef", "value_training", "prediction_trains_encoder", "target_ema", "conv_channels", "num_layers"} <= set(kind["fields"])
    assert kind["attention_view"] is None and not kind["recommended"]
    assert [m["key"] for m in kind["metrics"]] == ["prediction_loss", "embedding_std", "value_gap", "rollout_last_loss", "chained_value_gap"]
    assert all(m["label"] and m["description"] for m in kind["metrics"])
    assert [(m["setting"], m["above"]) for m in kind["metrics"]] == [("model.prediction_coef", 0.0)] * 3 + [("model.rollout_steps", 1.0)] * 2
    assert all(not row["metrics"] for name, row in schema["model_kinds"].items() if name not in ("jepa", "jepa_attention", "jepa_tokens", "jepa_temporal"))
    fields = {row["name"]: row for row in schema["sections"]["model"]}
    assert fields["value_training"]["choices"] == ["predicted", "actual"] and fields["value_training"]["description"]
    assert fields["prediction_trains_encoder"]["description"] and fields["prediction_trains_encoder"]["default"] is not None
    assert client.get("/api/defaults").json()["model"]["prediction_trains_encoder"] is False  # the New-training form's start value
    locked = client.get("/api/fork-rules").json()["locked"]
    objective = {"model.prediction_coef", "model.value_training", "model.target_ema", "model.prediction_trains_encoder"}
    assert "model.num_layers" in locked and not objective & set(locked)

    drawn = {algorithm: client.post("/api/architecture", json=tiny_tables("jepa", algorithm=algorithm)).json() for algorithm in ("ppo", "dqn")}
    assert all(word in drawn["ppo"]["text"] for word in ("predictor", "policy_decoder", "value_decoder"))
    assert "value_decoder" in drawn["dqn"]["text"] and "policy_decoder" not in drawn["dqn"]["text"]

    tables = tiny_tables("jepa", algorithm="dqn")
    tables["model"] |= {"value_training": "actual", "prediction_coef": 1.0}
    assert client.post("/api/runs", json={"name": "jepa", "config": tables}).status_code == 201
    detail = wait_until_done(client, "jepa")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["kind"] == "jepa" and {"prediction_loss", "embedding_std"} <= set(detail["metrics"][-1])

    watch = client.post("/api/watch", json={"run": "jepa"}).json()
    assert watch["info"]["attention_view"] is None and watch["frame"]["action_values"]["kind"] == "q_values"
    assert client.post(f"/api/watch/{watch['id']}/step", json={"count": 2}).status_code == 200

    tables["model"] |= {"prediction_coef": 0.5, "value_training": "predicted", "prediction_trains_encoder": False}  # the training objective may change on fork
    forked = client.post("/api/runs/jepa/fork", json={"name": "jepa-v2", "config": tables})
    assert forked.status_code == 201, forked.text
    forked_model = wait_until_done(client, "jepa-v2")["config"]["model"]
    assert forked_model["prediction_coef"] == 0.5 and forked_model["prediction_trains_encoder"] is False
    tables["model"]["num_layers"] = 2
    assert client.post("/api/runs/jepa/fork", json={"name": "jepa-v3", "config": tables}).status_code == 422

    ablation = tiny_tables("jepa", algorithm="dqn")
    ablation["model"]["prediction_coef"] = 0.0
    assert client.post("/api/runs", json={"name": "jepa-off", "config": ablation}).status_code == 201
    off = wait_until_done(client, "jepa-off")
    assert off["status"]["state"] == "finished" and "prediction_loss" not in off["metrics"][-1]


def test_rollout_steps_is_offered_for_the_jepa_kinds_only_reaches_a_run_and_may_change_on_fork(client: TestClient):
    schema = client.get("/api/schema").json()
    rollout = {"rollout_steps", "rollout_value_coef"}
    jepa_kinds = {"jepa", "jepa_attention", "jepa_tokens", "jepa_temporal"}
    assert {name for name, row in schema["model_kinds"].items() if rollout <= set(row["fields"])} == jepa_kinds
    assert not any(rollout & set(row["fields"]) for name, row in schema["model_kinds"].items() if name not in jepa_kinds)
    fields = {row["name"]: row for row in schema["sections"]["model"]}
    assert fields["rollout_steps"]["default"] == 1 and "its own prediction" in fields["rollout_steps"]["description"]
    assert fields["rollout_value_coef"]["default"] == 1.0 and fields["rollout_value_coef"]["description"]
    defaults = client.get("/api/defaults").json()["model"]
    assert defaults["rollout_steps"] == 1 and defaults["rollout_value_coef"] == 1.0  # the New-training form's start values
    assert not {"model.rollout_steps", "model.rollout_value_coef"} & set(client.get("/api/fork-rules").json()["locked"])

    tables = tiny_tables("jepa", algorithm="dqn")
    tables["model"] |= {"prediction_coef": 1.0, "rollout_steps": 3}
    assert client.post("/api/runs", json={"name": "rolled", "config": tables}).status_code == 201
    detail = wait_until_done(client, "rolled")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["config"]["model"]["rollout_steps"] == 3 and detail["config"]["model"]["rollout_value_coef"] == 1.0
    assert {"prediction_loss", "rollout_last_loss", "chained_value_gap"} <= set(detail["metrics"][-1])

    tables["model"] |= {"rollout_steps": 1}  # part of the training objective: a fork may switch it off again
    assert client.post("/api/runs/rolled/fork", json={"name": "unrolled", "config": tables}).status_code == 201
    unrolled = wait_until_done(client, "unrolled")
    assert unrolled["status"]["state"] == "finished", unrolled["log"]
    assert "prediction_loss" in unrolled["metrics"][-1] and not {"rollout_last_loss", "chained_value_gap"} & set(unrolled["metrics"][-1])

    refused = tiny_tables("jepa", algorithm="dqn")
    refused["model"]["rollout_steps"] = 0
    assert client.post("/api/runs", json={"name": "no-steps", "config": refused}).status_code == 422


def test_jepa_attention_is_offered_for_both_games_trained_and_watched_with_attention(client: TestClient):
    jepa = client.get("/api/schema").json()["model_kinds"]["jepa"]
    for game in ("snake", "lunar_lander"):
        kind = client.get("/api/schema", params={"game": game}).json()["model_kinds"]["jepa_attention"]
        assert {"num_heads", "num_layers", "conv_channels", "residual", "prediction_coef", "value_training"} <= set(kind["fields"])
        assert kind["attention_view"] == "grid" and not kind["recommended"] and kind["metrics"] == jepa["metrics"]
        assert "num_layers sets both" in kind["description"]

    drawn = client.post("/api/architecture", json=tiny_tables("jepa_attention", algorithm="dqn")).json()
    assert all(word in drawn["text"] for word in ("TransformerBlock", "predictor", "value_decoder"))

    tables = tiny_tables("jepa_attention", algorithm="dqn")
    tables["model"]["prediction_coef"] = 1.0
    assert client.post("/api/runs", json={"name": "jepa-attention", "config": tables}).status_code == 201
    detail = wait_until_done(client, "jepa-attention")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["kind"] == "jepa_attention" and {"prediction_loss", "embedding_std", "value_gap"} <= set(detail["metrics"][-1])

    watch = client.post("/api/watch", json={"run": "jepa-attention"}).json()
    assert watch["info"]["attention_view"] == "grid" and watch["frame"]["action_values"]["kind"] == "q_values"
    assert len(watch["frame"]["heatmap"]) == 4
    steps = client.post(f"/api/watch/{watch['id']}/step", json={"count": 2})
    assert steps.status_code == 200 and all("heatmap" in frame for frame in steps.json()["frames"])


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


@pytest.mark.parametrize("game", ["snake", "lunar_lander"])
def test_residual_is_offered_for_the_kinds_with_a_stem_and_locked_on_fork(client: TestClient, game: str):
    schema = client.get("/api/schema", params={"game": game}).json()
    assert {kind for kind, info in schema["model_kinds"].items() if "residual" in info["fields"]} == {"hybrid", "jepa_attention", "jepa_tokens", "frame_attention", "lstm"}
    field = next(row for row in schema["sections"]["model"] if row["name"] == "residual")
    assert field["type"] == "bool" and field["default"] is False and field["description"]
    assert "model.residual" in client.get("/api/fork-rules", params={"game": game}).json()["locked"]


def test_a_residual_hybrid_trains_is_drawn_and_cannot_change_on_fork(client: TestClient):
    tables = tiny_tables("hybrid")
    tables["model"]["residual"] = True
    drawn = client.post("/api/architecture", json=tables).json()
    assert "ResidualConvBlock" in drawn["text"]
    assert client.post("/api/runs", json={"name": "res", "config": tables}).status_code == 201
    detail = wait_until_done(client, "res")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["config"]["model"]["residual"] is True
    tables["model"]["residual"] = False
    refused = client.post("/api/runs/res/fork", json={"name": "res-v2", "config": tables})
    assert refused.status_code == 422 and "model.residual" in refused.text


LAYER_NORM_KINDS = {"cnn", "resnet", "mlp", "jepa", "hybrid", "jepa_attention", "jepa_tokens", "frame_attention", "lstm"}


@pytest.mark.parametrize("game", ["snake", "lunar_lander"])
def test_layer_norm_and_weight_decay_are_in_the_schema(client: TestClient, game: str):
    schema = client.get("/api/schema", params={"game": game}).json()
    assert {kind for kind, info in schema["model_kinds"].items() if "layer_norm" in info["fields"]} == LAYER_NORM_KINDS
    layer_norm = next(row for row in schema["sections"]["model"] if row["name"] == "layer_norm")
    assert layer_norm["type"] == "bool" and layer_norm["default"] is False and "group norm" in layer_norm["description"]
    weight_decay = next(row for row in schema["sections"]["train"] if row["name"] == "weight_decay")
    assert weight_decay["type"] == "float" and weight_decay["default"] == 0.0 and "AdamW" in weight_decay["description"]
    locked = client.get("/api/fork-rules", params={"game": game}).json()["locked"]
    assert "model.layer_norm" in locked and "train.weight_decay" not in locked
    for algorithm in ("ppo", "dqn") if game == "snake" else ():  # the defaults the forms are filled with leave both off (this fixture has Snake's files only)
        defaults = client.get("/api/defaults", params={"game": game, "algorithm": algorithm}).json()
        assert defaults["model"]["layer_norm"] is False and defaults["train"]["weight_decay"] == 0.0


def test_the_parameter_count_reflects_layer_norm_for_the_kinds_that_read_it(client: TestClient):
    for kind in ("cnn", "mlp", "hybrid", "transformer"):
        tables = tiny_tables(kind)
        plain = parameters_for(client, tables)
        tables["model"]["layer_norm"] = True
        normed = parameters_for(client, tables)
        assert (normed > plain) == (kind in LAYER_NORM_KINDS), kind
        tables["train"]["weight_decay"] = 0.01  # an optimizer setting: no parameters
        assert parameters_for(client, tables) == normed
    cnn = tiny_tables("cnn")  # one conv (8 channels) and the readout (16 wide): a scale and a shift each
    cnn["model"]["layer_norm"] = True
    assert parameters_for(client, cnn) - parameters_for(client, tiny_tables("cnn")) == 2 * 8 + 2 * 16


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_a_run_with_layer_norm_and_weight_decay_trains_is_drawn_and_forks_with_another_weight_decay(client: TestClient, algorithm: str):
    tables = tiny_tables("cnn", algorithm=algorithm, weight_decay=0.01)
    tables["model"]["layer_norm"] = True
    drawn = client.post("/api/architecture", json=tables).json()
    assert "GroupNorm" in drawn["text"] and "LayerNorm" in drawn["text"]
    assert client.post("/api/runs", json={"name": "normed", "config": tables}).status_code == 201
    detail = wait_until_done(client, "normed")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["config"]["model"]["layer_norm"] is True and detail["config"]["train"]["weight_decay"] == 0.01
    assert "GroupNorm" in client.get("/api/runs/normed/architecture").json()["text"]

    for name, weight_decay in (("normed-off", 0.0), ("normed-more", 0.1)):  # AdamW -> Adam, and another strength
        tables["train"]["weight_decay"] = weight_decay
        forked = client.post("/api/runs/normed/fork", json={"name": name, "config": tables})
        assert forked.status_code == 201, forked.text
        child = wait_until_done(client, name)
        assert child["status"]["state"] == "finished", child["log"]
        assert child["config"]["train"]["weight_decay"] == weight_decay and child["parent"]["run"] == "normed"

    tables["model"]["layer_norm"] = False
    refused = client.post("/api/runs/normed/fork", json={"name": "normed-plain", "config": tables})
    assert refused.status_code == 422 and "model.layer_norm" in refused.text


def test_jepa_tokens_is_offered_for_both_games_trained_and_watched_with_attention(client: TestClient):
    jepa = client.get("/api/schema").json()["model_kinds"]["jepa"]
    for game in ("snake", "lunar_lander"):
        schema = client.get("/api/schema", params={"game": game}).json()
        kind = schema["model_kinds"]["jepa_tokens"]
        assert {"num_heads", "num_layers", "predictor_layers", "residual", "prediction_coef", "value_training"} <= set(kind["fields"])
        assert kind["attention_view"] == "grid" and not kind["recommended"] and kind["metrics"] == jepa["metrics"]
        assert all("predictor_layers" not in row["fields"] for name, row in schema["model_kinds"].items() if name != "jepa_tokens")
    assert "model.predictor_layers" in client.get("/api/fork-rules").json()["locked"]

    tables = tiny_tables("jepa_tokens", algorithm="dqn")
    tables["model"]["prediction_coef"] = 1.0
    assert client.post("/api/runs", json={"name": "jepa-tokens", "config": tables}).status_code == 201
    detail = wait_until_done(client, "jepa-tokens")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["kind"] == "jepa_tokens" and {"prediction_loss", "embedding_std", "value_gap"} <= set(detail["metrics"][-1])

    watch = client.post("/api/watch", json={"run": "jepa-tokens"}).json()
    assert watch["info"]["attention_view"] == "grid" and watch["frame"]["action_values"]["kind"] == "q_values"
    assert len(watch["frame"]["heatmap"]) == 4 and len(watch["frame"]["action_values"]["values"]) == 4
    steps = client.post(f"/api/watch/{watch['id']}/step", json={"count": 2})
    assert steps.status_code == 200 and all("heatmap" in frame for frame in steps.json()["frames"])


def test_jepa_temporal_is_offered_for_both_games_with_history_trained_and_watched_with_frame_attention(client: TestClient, paths: AppPaths):
    jepa = client.get("/api/schema").json()["model_kinds"]["jepa"]
    for game in ("snake", "lunar_lander"):
        kind = client.get("/api/schema", params={"game": game}).json()["model_kinds"]["jepa_temporal"]
        assert {"num_heads", "num_layers", "prediction_coef", "value_training", "target_ema"} <= set(kind["fields"])
        assert not {"conv_channels", "residual", "predictor_layers"} & set(kind["fields"])
        assert kind["uses_history"] and kind["recommended_env"] == {"history_length": 4, "history_actions": True}
        assert kind["attention_view"] == "frames" and not kind["recommended"] and kind["metrics"] == jepa["metrics"]
        assert "num_layers sets both" in kind["description"]

    tables = tiny_tables("jepa_temporal")
    tables["model"]["prediction_coef"] = 1.0
    tables["env"]["history_actions"] = True
    assert client.post("/api/runs", json={"name": "jepa-temporal", "config": tables}).status_code == 201
    detail = wait_until_done(client, "jepa-temporal")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["kind"] == "jepa_temporal" and {"prediction_loss", "embedding_std", "value_gap"} <= set(detail["metrics"][-1])
    drawn = client.post("/api/architecture", json=tables).json()
    assert all(word in drawn["text"] for word in ("TransformerBlock", "TemporalEmbedding", "predictor", "value_decoder"))

    watch = client.post("/api/watch", json={"run": "jepa-temporal"}).json()
    assert watch["info"]["attention_view"] == "frames" and watch["info"]["history_length"] == 2
    assert watch["frame"]["action_values"]["kind"] == "probabilities" and len(watch["frame"]["heatmap"]) == 4
    steps = client.post(f"/api/watch/{watch['id']}/step", json={"count": 2})
    assert steps.status_code == 200 and all("heatmap" in frame for frame in steps.json()["frames"])

    make_run(paths, "blind", kind="jepa_temporal", game={"name": "lunar_lander", "hide_velocities": True})
    body = client.post("/api/watch", json={"game": "lunar_lander", "run": "blind", "seed": 0}).json()
    assert body["info"]["attention_view"] == "frames" and len(body["info"]["element_names"]) == 8
    frame = client.post(f"/api/watch/{body['id']}/step", json={"count": 2, "frames": "last"}).json()["frames"][-1]
    assert len(frame["element_weights"]) == 8 and frame["heatmap"] is None


# --------------------------------------------------------------------------- the training queue

LONG = 10_000_000  # env steps of a run that trains until it is stopped


@pytest.fixture
def queue_client(paths: AppPaths):
    """A client whose app looks at the training queue often, so tests do not wait for it."""
    with TestClient(create_app(paths, queue_interval=0.05)) as test_client:
        yield test_client


def enqueue(client: TestClient, name: str, total_env_steps: int = LONG, **body):
    return client.post("/api/queue/runs", json={"name": name, "config": tiny_tables(total_env_steps=total_env_steps)} | body)


def queue_runs(client: TestClient) -> tuple[list[str], list[str]]:
    """(runs of the entries in the queue, in order; runs started from the queue that still train, sorted)."""
    listed = client.get("/api/queue").json()
    return [entry["run"] for entry in listed["entries"]], sorted(job["run"] for job in listed["running"])


def wait_for_queue(client: TestClient, entries: list[str], running: list[str]) -> None:
    deadline = time.monotonic() + JOB_TIMEOUT
    while queue_runs(client) != (entries, running):
        assert time.monotonic() < deadline, f"queue is {queue_runs(client)}, expected {(entries, running)}"
        time.sleep(0.05)


def test_queued_runs_start_one_after_another_and_their_names_are_reserved(queue_client: TestClient, paths: AppPaths):
    client = queue_client
    assert client.get("/api/queue").json() == {"max_parallel": 1, "entries": [], "running": []}
    first = enqueue(client, "a", description="  first  ")
    assert first.status_code == 201, first.text
    assert first.json() | {"id": "", "queued_at": 0} == {
        "id": "", "game": "snake", "run": "a", "kind": "new", "parent": None, "checkpoint": None, "description": "first",
        "queued_at": 0, "state": "queued", "error": None, "model_kind": "transformer", "algorithm": "ppo",
        "total_env_steps": LONG, "outcome": "running",  # nothing was running: it started at once
    }
    assert enqueue(client, "b").json()["outcome"] == "queued"
    assert enqueue(client, "c").status_code == 201
    assert queue_runs(client) == (["b", "c"], ["a"])
    assert not (paths.runs_dir / "snake" / "b").exists()  # a queued run has no directory yet

    # A started entry is an ordinary run.
    started = client.get("/api/runs/a").json()
    assert started["status"]["state"] == "running" and started["status"]["mode"] == "new"
    assert started["config"]["train"]["total_env_steps"] == LONG

    # Without HTTP: what became of an entry.
    launcher = client.app.state.launcher
    outcome = launcher.outcome(first.json()["id"])
    assert outcome.state == "running" and outcome.run_dir == paths.runs_dir / "snake" / "a" and not outcome.done

    # A queued name is taken, like an existing run's.
    make_run(paths, "parent")
    assert client.post("/api/runs", json={"name": "b", "config": tiny_tables()}).status_code == 409
    assert client.post("/api/runs/parent/fork", json={"name": "b", "config": tiny_tables()}).status_code == 409
    assert enqueue(client, "b").status_code == 409
    assert enqueue(client, "a").status_code == 409  # the run exists
    assert enqueue(client, "parent").status_code == 409

    # Validated when queued, as a start would.
    assert enqueue(client, "../escape").status_code == 422
    tables = tiny_tables()
    tables["train"]["learning_rate"] = -1.0
    bad = client.post("/api/queue/runs", json={"name": "bad", "config": tables})
    assert bad.status_code == 422 and "learning_rate" in bad.json()["detail"]
    assert queue_runs(client) == (["b", "c"], ["a"])

    assert client.post("/api/runs/a/stop").status_code == 200
    wait_for_queue(client, ["c"], ["b"])
    assert client.get("/api/runs/a").json()["status"]["state"] == "stopped"
    assert client.post("/api/runs/b/stop").json()["status"]["state"] == "stopped"  # the usual stop button
    wait_for_queue(client, [], ["c"])
    assert launcher.outcome(first.json()["id"]).state == "stopped" and launcher.outcome(first.json()["id"]).done


def test_queue_runs_max_parallel_at_once(queue_client: TestClient):
    client = queue_client
    assert client.get("/api/queue/settings").json() == {"max_parallel": 1}
    assert client.put("/api/queue/settings", json={"max_parallel": 2}).json() == {"max_parallel": 2}
    assert client.put("/api/queue/settings", json={"max_parallel": -1}).status_code == 422
    for name in "abc":
        assert enqueue(client, name).status_code == 201
    assert queue_runs(client) == (["c"], ["a", "b"])
    client.post("/api/runs/a/stop")
    wait_for_queue(client, [], ["b", "c"])


def test_run_now_jumps_the_queue_and_an_entry_can_be_removed(queue_client: TestClient, paths: AppPaths):
    client = queue_client
    ids = {name: enqueue(client, name).json()["id"] for name in "abcd"}
    assert queue_runs(client) == (["b", "c", "d"], ["a"])
    now = client.post(f"/api/queue/{ids['c']}/run")
    assert now.status_code == 200, now.text
    assert now.json()["name"] == "c" and now.json()["game"] == "snake" and now.json()["status"]["state"] == "running"
    assert queue_runs(client) == (["b", "d"], ["a", "c"])
    assert client.post(f"/api/queue/{ids['c']}/run").status_code == 404  # no longer queued

    removed = client.delete(f"/api/queue/{ids['d']}")
    assert removed.status_code == 200 and removed.json()["run"] == "d"
    assert client.delete(f"/api/queue/{ids['d']}").status_code == 404
    assert client.delete("/api/queue/no-such-entry").status_code == 404
    assert queue_runs(client) == (["b"], ["a", "c"]) and not (paths.runs_dir / "snake" / "d").exists()
    again = enqueue(client, "d")  # the name is free again
    assert again.status_code == 201 and client.delete(f"/api/queue/{again.json()['id']}").status_code == 200

    client.post("/api/runs/a/stop")  # c was started by hand, so it holds no slot: b starts while c trains
    wait_for_queue(client, [], ["b", "c"])


def test_a_queued_fork_that_cannot_start_is_kept_as_failed_and_the_queue_goes_on(queue_client: TestClient, paths: AppPaths):
    client = queue_client
    parent = make_run(paths, "parent")
    client.put("/api/queue/settings", json={"max_parallel": 0})  # paused: entries wait

    def fork(parent_name: str = "parent", **body):
        body = {"name": "child", "checkpoint": "checkpoint_0000000016.pt", "config": tiny_tables(total_env_steps=32)} | body
        return client.post(f"/api/queue/runs/{parent_name}/fork", json=body)

    assert fork("missing").status_code == 404
    assert fork(checkpoint="checkpoint_0000000099.pt").status_code == 404
    locked = tiny_tables()
    locked["model"]["d_model"] = 32
    refused = fork(config=locked)
    assert refused.status_code == 422 and "model.d_model" in refused.json()["detail"]
    queued = fork()
    assert queued.status_code == 201, queued.text
    entry = queued.json()
    assert (entry["kind"], entry["parent"], entry["checkpoint"], entry["outcome"]) == ("fork", "parent", "checkpoint_0000000016.pt", "queued")
    assert enqueue(client, "after").json()["outcome"] == "queued"

    (parent / "checkpoint_0000000016.pt").unlink()  # the parent's checkpoint goes away while the fork waits
    client.put("/api/queue/settings", json={"max_parallel": 1})
    wait_for_queue(client, ["child"], ["after"])
    failed = client.get("/api/queue").json()["entries"][0]
    assert failed["state"] == "failed" and "checkpoint_0000000016.pt" in failed["error"]
    assert not (paths.runs_dir / "snake" / "child").exists()
    outcome = client.app.state.launcher.outcome(entry["id"])
    assert outcome.state == "failed_to_start" and outcome.done and "checkpoint" in outcome.error

    retried = client.post(f"/api/queue/{entry['id']}/run")  # still impossible: the reason is the answer
    assert retried.status_code == 404 and "checkpoint_0000000016.pt" in retried.json()["detail"]
    assert client.get("/api/queue").json()["entries"][0]["state"] == "failed"
    assert enqueue(client, "child").status_code == 201  # a failed entry does not hold its name
    assert client.delete(f"/api/queue/{entry['id']}").status_code == 200
    assert queue_runs(client) == (["child"], ["after"])


def test_the_queue_survives_a_restart_and_a_queued_fork_and_resume_train(paths: AppPaths):
    make_run(paths, "parent")
    with TestClient(create_app(paths, queue_interval=0.05)) as first:
        assert enqueue(first, "long").json()["outcome"] == "running"
        fork = {"name": "child", "config": tiny_tables(total_env_steps=32), "description": "queued fork"}
        assert first.post("/api/queue/runs/parent/fork", json=fork).status_code == 201
        resumed = first.post("/api/queue/runs/parent/resume", json={"total_env_steps": 48})
        assert resumed.status_code == 201, resumed.text
        assert (resumed.json()["kind"], resumed.json()["total_env_steps"], resumed.json()["model_kind"]) == ("resume", 48, None)
        assert first.post("/api/queue/runs/parent/resume", json={}).status_code == 409  # one entry per run
        assert first.post("/api/queue/runs/missing/resume", json={}).status_code == 404
        assert queue_runs(first) == (["child", "parent"], ["long"])
    # The server stopped its job without starting the next entry; the queue is on disk.
    saved = json.loads((paths.runs_dir / "queue.json").read_text())
    assert [entry["request"]["run"] for entry in saved["entries"]] == ["child", "parent"] and saved["max_parallel"] == 1
    assert not (paths.runs_dir / "snake" / "child").exists()

    with TestClient(create_app(paths, queue_interval=0.05)) as second:  # the queue goes on by itself
        wait_for_queue(second, [], [])
        assert second.get("/api/runs/long").json()["status"] == {"state": "idle"}  # jobs die with their server
        child = wait_until_done(second, "child")
        assert child["status"]["state"] == "finished" and child["status"]["mode"] == "fork", child["log"]
        assert child["checkpoints"][-1]["steps"] == 32 and child["description"] == "queued fork"
        assert child["parent"] == {"run": "parent", "checkpoint": "checkpoint_0000000016.pt", "env_steps": 16}
        assert [event["action"] for event in child["history"]] == ["fork"]
        parent = wait_until_done(second, "parent")
        assert parent["status"]["state"] == "finished" and parent["status"]["mode"] == "resume", parent["log"]
        assert parent["checkpoints"][-1]["steps"] == 48 and parent["total_env_steps"] == 48
        assert json.loads((paths.runs_dir / "queue.json").read_text())["entries"] == []


def test_the_queue_is_one_for_all_games(multi_client: TestClient, multi_paths: AppPaths):
    client = multi_client
    client.put("/api/queue/settings", json={"max_parallel": 0})
    tables = tiny_tables(total_env_steps=32)
    tables["game"] = BEACON
    beacon = client.post("/api/queue/runs", json={"name": "tiny", "config": tables})
    assert beacon.status_code == 201 and beacon.json()["game"] == "beacon"
    assert enqueue(client, "tiny").json()["game"] == "snake"  # the same name in another game is another run
    assert [(e["game"], e["run"]) for e in client.get("/api/queue").json()["entries"]] == [("beacon", "tiny"), ("snake", "tiny")]
    assert [e["game"] for e in client.get("/api/queue", params={"game": "beacon"}).json()["entries"]] == ["beacon"]
    assert [e["game"] for e in client.get("/api/queue", params={"game": "snake"}).json()["entries"]] == ["snake"]
    assert client.get("/api/queue", params={"game": "chess"}).status_code == 404
    assert client.post("/api/queue/runs/tiny/resume", json={}, params={"game": "chess"}).status_code == 404
    assert client.get("/api/runs", params={"game": "beacon"}).json() == []  # nothing started, nothing created
    assert client.get("/api/games").status_code == 200  # the queue file in the runs root is not mistaken for a game
    assert (multi_paths.runs_dir / "queue.json").is_file()


# --------------------------------------------------------------------------- reward search


def wait_for_search(client: TestClient, name: str) -> dict:
    deadline = time.monotonic() + 2 * JOB_TIMEOUT
    while time.monotonic() < deadline:
        detail = client.get(f"/api/reward-search/{name}").json()
        if detail["status"] != "running":
            return detail
        time.sleep(0.2)
    raise AssertionError(f"reward search {name} still running: {client.get(f'/api/reward-search/{name}').json()}")


def test_the_reward_search_form_offers_coefficients_and_real_objectives_only(client: TestClient):
    form = client.get("/api/reward-search/form").json()
    assert form["game"] == "snake" and "step" in [row["name"] for row in form["fields"]]
    rows = {row["name"]: row for row in form["settings"]}
    assert {"population", "generations", "steps_per_member", "perturbation", "max_change", "zero_scale", "allow_sign_change", "seed"} <= set(rows)
    assert all(row["description"] for row in form["settings"])
    assert rows["objective"]["default"] == "mean_score" and "mean_return" not in rows["objective"]["choices"]
    atari = client.get("/api/reward-search/form", params={"game": "atari"}).json()
    assert [row["name"] for row in atari["fields"]] == ["scale", "life_lost"]
    assert client.get("/api/reward-search/form", params={"game": "chess"}).status_code == 404


def test_a_reward_search_runs_through_the_queue_and_its_members_are_ordinary_runs(queue_client: TestClient, paths: AppPaths):
    client = queue_client
    make_run(paths, "parent")
    body = {
        "name": "rs", "parent": "parent", "fields": ["death", "step"],
        "settings": {"population": 2, "generations": 2, "steps_per_member": 32, "evaluation_games": 2, "seed": 1},
    }
    assert client.post("/api/reward-search", json=body | {"parent": "missing"}).status_code == 404
    assert client.post("/api/reward-search", json=body | {"checkpoint": "nope.pt"}).status_code == 404
    assert client.post("/api/reward-search", json=body | {"fields": ["nope"]}).status_code == 422
    assert client.post("/api/reward-search", json=body | {"fields": []}).status_code == 422
    assert client.post("/api/reward-search", json=body | {"name": "../x"}).status_code == 422
    assert client.post("/api/reward-search", json=body | {"name": "form"}).status_code == 422
    assert client.post("/api/reward-search", json=body | {"settings": {"population": 1}}).status_code == 422
    assert client.post("/api/reward-search", json=body | {"settings": {"objective": "mean_return"}}).status_code == 422
    assert client.post("/api/reward-search", json=body | {"settings": {"unknown": 1}}).status_code == 422
    started = client.post("/api/reward-search", json=body)
    assert started.status_code == 201, started.text
    assert started.json()["status"] == "running" and started.json()["state"]["spec"]["checkpoint"] == "checkpoint_0000000016.pt"
    assert client.post("/api/reward-search", json=body).status_code == 409

    detail = wait_for_search(client, "rs")
    assert detail["status"] == "finished", detail
    generations = detail["state"]["generations"]
    assert [[m["run"] for m in g["members"]] for g in generations] == [["rs-g0-m0", "rs-g0-m1"], ["rs-g1-m0", "rs-g1-m1"]]
    assert all(m["state"] == "done" and m["score"] is not None for g in generations for m in g["members"])
    first_winner = generations[0]["members"][generations[0]["winner"]]
    assert generations[1]["parent"] == first_winner["run"] and generations[1]["members"][0]["reward"] == first_winner["reward"]
    assert detail["generations_judged"] == 2 and detail["best_run"].startswith("rs-g1-m")

    runs = {run["name"]: run for run in client.get("/api/runs").json()}
    assert {"rs-g0-m0", "rs-g0-m1", "rs-g1-m0", "rs-g1-m1"} <= set(runs)
    member = client.get("/api/runs/rs-g1-m1").json()
    assert member["parent"]["run"] == first_winner["run"] and member["config"]["reward"] == generations[1]["members"][1]["reward"]
    assert "Reward search rs, generation 1, member 1" in member["description"]
    assert member["evaluations"][-1]["statistics"]["games"] == 2
    assert [s["name"] for s in client.get("/api/reward-search", params={"game": "snake"}).json()] == ["rs"]
    assert client.get("/api/reward-search", params={"game": "lunar_lander"}).json() == []
    assert client.get("/api/reward-search/missing").status_code == 404
    assert client.post("/api/reward-search/rs/stop").json()["status"] == "finished"  # nothing to stop
    assert client.get("/api/games").status_code == 200 and (paths.runs_dir / "reward_searches" / "rs.json").is_file()


def test_stopping_a_reward_search_empties_its_queue_entries_and_survives_a_restart(paths: AppPaths):
    make_run(paths, "parent")
    body = {"name": "halt", "parent": "parent", "fields": ["food"], "settings": {"population": 3, "steps_per_member": 100_000}}
    with TestClient(create_app(paths, queue_interval=0.05)) as client:
        client.put("/api/queue/settings", json={"max_parallel": 0})  # members wait in the queue
        assert client.post("/api/reward-search", json=body).status_code == 201
        deadline = time.monotonic() + 30
        while len(client.get("/api/queue").json()["entries"]) < 3:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    with TestClient(create_app(paths, queue_interval=0.05)) as client:  # after a restart: nothing queued twice
        time.sleep(0.5)
        assert [e["run"] for e in client.get("/api/queue").json()["entries"]] == ["halt-g0-m0", "halt-g0-m1", "halt-g0-m2"]
        assert {m["state"] for m in client.get("/api/reward-search/halt").json()["state"]["generations"][0]["members"]} == {"waiting"}
        stopped = client.post("/api/reward-search/halt/stop").json()
        assert stopped["status"] == "stopped" and client.get("/api/queue").json()["entries"] == []


# --------------------------------------------------------------------------- Atari (image observations)


def tiny_atari(kind: str = "cnn", algorithm: str = "ppo") -> dict:
    """Pong on a 48 x 48 picture, games cut after 6 steps: trains in a few seconds on the CPU."""
    tables = tiny_tables(kind, total_env_steps=32, algorithm=algorithm)
    tables["game"] = {"name": "atari", "title": "pong", "screen_size": 48, "max_episode_steps": 6}
    tables["env"]["history_length"] = 2
    return tables


def wait_for_evaluation_of(client: TestClient, name: str, game: str) -> dict:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{name}/evaluate", params={"game": game}).json()
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"evaluation of {name} still running after {JOB_TIMEOUT}s")


def test_atari_is_listed_with_its_titles_and_the_kinds_that_read_pictures(client: TestClient):
    games = {game["name"]: game for game in client.get("/api/games").json()}
    atari = games["atari"]
    assert (atari["title"], atari["observation_kind"], atari["view_kind"], atari["has_reward_table"]) == ("Atari", "image", "image", False)
    assert [metric["key"] for metric in atari["evaluation_metrics"]][:3] == ["timeout_rate", "mean_lives_lost", "mean_scoring_steps"]
    schema = client.get("/api/schema", params={"game": "atari"}).json()
    title = next(row for row in schema["sections"]["game"] if row["name"] == "title")
    assert {"pong", "montezuma_revenge", "ms_pacman", "pitfall", "private_eye", "venture"} <= set(title["choices"])
    assert "cnn" in schema["model_kinds"] and not {"resnet", "mlp"} & set(schema["model_kinds"])  # not offered: they cannot read pictures
    assert set(client.get("/api/schema").json()["model_kinds"]) >= {"resnet", "mlp"}
    locked = client.get("/api/fork-rules", params={"game": "atari"}).json()["locked"]
    assert {"game.title", "game.screen_size", "game.grayscale", "game.full_action_space"} <= set(locked) and "game.frame_skip" not in locked
    assert parameters_for(client, tiny_atari(), game="atari") > 0


def test_train_watch_evaluate_and_compare_atari_through_the_api(client: TestClient, paths: AppPaths):
    """The real emulator in a real training subprocess, then everything the tabs ask for."""
    tables = tiny_atari()
    tables["evaluation"] = {"interval": 32, "games": 2}
    started = client.post("/api/runs", json={"name": "pong", "config": tables})
    assert started.status_code == 201, started.text
    assert started.json()["game"] == "atari"
    detail = wait_until_done(client, "pong", game="atari")
    assert detail["status"]["state"] == "finished", detail["log"]
    assert detail["rules"]["title"] == "pong" and detail["kind"] == "cnn" and detail["checkpoints"][-1]["steps"] == 32
    assert (paths.runs_dir / "atari" / "pong" / "config.toml").is_file()
    recorded = detail["evaluations"][-1]
    assert recorded["statistics"]["games"] == 2 and recorded["statistics"]["timeout_rate"] == 1.0  # cut after 6 steps
    assert "DownsamplingConvEncoder" in client.get("/api/runs/pong/architecture", params={"game": "atari"}).json()["text"]

    watch = client.post("/api/watch", json={"game": "atari", "run": "pong", "seed": 0})
    assert watch.status_code == 201, watch.text
    body = watch.json()
    assert (body["info"]["view_kind"], body["info"]["width"], body["info"]["height"]) == ("image", 160, 210)
    assert len(body["info"]["action_names"]) == 18 and body["info"]["action_names"][:2] == ["nothing", "fire"]
    assert base64.b64decode(body["frame"]["png"]).startswith(b"\x89PNG")
    assert body["frame"]["action_values"]["kind"] == "probabilities" and len(body["frame"]["action_values"]["values"]) == 18
    steps = client.post(f"/api/watch/{body['id']}/step", json={"count": 8, "frames": "last"}).json()["frames"]
    assert "png" in steps[-1] and any(frame["event"] == "timeout" for frame in steps)

    assert client.post("/api/runs/pong/evaluate", params={"game": "atari"}, json={"games": 2, "seed": 1000}).status_code == 202
    status = wait_for_evaluation_of(client, "pong", "atari")
    assert status["state"] == "finished", status
    assert status["result"]["evaluation"]["statistics"] == recorded["statistics"]  # seeded: the training-time numbers again
    assert status["result"]["reproduces"]["identical"] is True

    make_run(paths, "eyes", kind="hybrid", game=tables["game"])
    compared = client.post("/api/compare", json={"game": "atari", "runs": ["pong", "eyes"]}).json()
    assert {run["config"]["game"]["title"] for run in compared["runs"]} == {"pong"}
    assert ("model", "kind") in {(d["section"], d["field"]) for d in compared["differences"]}
    eyes = client.post("/api/watch", json={"game": "atari", "run": "eyes", "seed": 0}).json()
    assert [len(eyes["frame"]["heatmap"]), len(eyes["frame"]["heatmap"][0])] == [2, 2]  # the stem's 2 x 2 feature map of a 48 x 48 picture

    other_title = {**tables, "game": {**tables["game"], "title": "venture"}}
    refused = client.post("/api/runs/pong/fork", params={"game": "atari"}, json={"name": "pong-v2", "config": other_title})
    assert refused.status_code == 422 and "game.title" in refused.text


def test_atari_trains_with_dqn_through_the_api(client: TestClient):
    started = client.post("/api/runs", json={"name": "pong-dqn", "config": tiny_atari(algorithm="dqn")})
    assert started.status_code == 201, started.text
    detail = wait_until_done(client, "pong-dqn", game="atari")
    assert detail["status"]["state"] == "finished", detail["log"]
    watch = client.post("/api/watch", json={"game": "atari", "run": "pong-dqn"}).json()
    assert watch["frame"]["action_values"]["kind"] == "q_values" and len(watch["frame"]["action_values"]["values"]) == 18
