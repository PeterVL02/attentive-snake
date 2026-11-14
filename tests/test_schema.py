import dataclasses
import json

import pytest
import torch

from attentive_snake.config import ExperimentConfig
from attentive_snake.config.experiment import section_types
from attentive_snake.config.schema import experiment_schema
from attentive_snake.training.algorithms import algorithm_names
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.observation_spec import ObservationSpec

SCHEMA = experiment_schema()


def test_sections_in_order_and_cover_every_dataclass_field() -> None:
    assert SCHEMA["game"] == "snake"
    assert list(SCHEMA["sections"]) == ["game", "reward", "model", "train", "env", "evaluation", *algorithm_names()]
    for section, rows in SCHEMA["sections"].items():
        fields = [f.name for f in dataclasses.fields(section_types("snake", section if section in algorithm_names() else "ppo")[section])]
        assert [row["name"] for row in rows] == fields
    assert "name" not in [row["name"] for row in SCHEMA["sections"]["game"]]  # [game] name is the top-level "game"


def test_schema_of_an_unknown_game_fails() -> None:
    with pytest.raises(ValueError, match="unknown game"):
        experiment_schema("no_such_game")


def test_every_field_has_a_description_and_the_documented_row_shape() -> None:
    for rows in SCHEMA["sections"].values():
        for row in rows:
            assert set(row) == {"name", "type", "default", "nullable", "description", "choices"}
            assert row["description"].strip(), row["name"]
            assert row["type"] in {"int", "float", "bool", "str", "path"}


def test_schema_is_json_serialisable() -> None:
    assert json.loads(json.dumps(SCHEMA)) == SCHEMA


def test_nullable_and_choices_and_defaults() -> None:
    rows = {(s, r["name"]): r for s, rs in SCHEMA["sections"].items() for r in rs}
    assert rows["game", "max_steps_without_food"]["nullable"] and rows["game", "max_steps_without_food"]["default"] is None
    assert rows["train", "checkpoint_dir"]["type"] == "path"
    assert rows["ppo", "target_kl"]["type"] == "float"
    assert not rows["train", "gamma"]["nullable"] and rows["train", "gamma"]["default"] == 0.99
    assert rows["model", "kind"]["choices"] == list(POLICY_KINDS)
    assert rows["model", "d_model"]["choices"] is None


def test_toml_round_trip_still_works_with_metadata(tmp_path) -> None:
    from attentive_snake.config import load_experiment, save_experiment

    save_experiment(ExperimentConfig(), tmp_path / "c.toml")
    assert load_experiment(tmp_path / "c.toml") == ExperimentConfig()


def test_model_kinds_follow_registry_order_with_documented_keys() -> None:
    assert list(SCHEMA["model_kinds"]) == list(POLICY_KINDS)
    model_fields = {f.name for f in dataclasses.fields(ModelConfig)} - {"kind"}
    for kind, info in SCHEMA["model_kinds"].items():
        assert set(info) == {"description", "observation_kinds", "recommended", "fields", "uses_history", "recommended_env", "attention_view"}
        assert info["description"].strip()
        assert info["observation_kinds"] == list(KIND_INFO[kind].observation_kinds) and "grid" in info["observation_kinds"]
        assert info["description"] == KIND_INFO[kind].inputs["grid"].description
        assert set(info["fields"]) <= model_fields
        temporal = info["uses_history"]
        assert info["recommended_env"] == {"history_length": 4 if temporal else 1, "history_actions": temporal}
        assert info["attention_view"] in ("grid", "frames", "weights", None)
    assert {k for k, i in SCHEMA["model_kinds"].items() if i["uses_history"]} == {"temporal_transformer", "frame_attention", "lstm"}
    assert SCHEMA["model_kinds"]["cnn"]["attention_view"] is None
    assert {k for k, i in SCHEMA["model_kinds"].items() if not i["recommended"]} == {"mlp", "lstm"}  # Snake is a grid game


def test_every_vector_reading_names_only_listed_fields() -> None:
    for info in KIND_INFO.values():
        for reading in info.inputs.values():
            assert set(reading.unused_fields) <= set(info.fields) and reading.description.strip()


# Fields that change the parameter count when used. num_heads and dropout never do
# (heads split d_model; dropout has no weights), so they are checked through outputs below.
SIZE_CHANGES = {
    "d_model": 32,
    "num_layers": 2,
    "mlp_ratio": 2,
    "conv_channels": 24,
    "conv_layers": 2,
    "kernel_size": 5,
}


def _build(config: ModelConfig, observation: str = "grid"):
    """A policy for a 5x5 board or 6 numbers; temporal kinds get 2 frames with action channels (4 per frame)."""
    temporal = KIND_INFO[config.kind].uses_history
    channels = (3 if observation == "grid" else 6) + (4 if temporal else 0)
    frames = 2 if temporal else 1
    shape = (frames * channels, 5, 5) if observation == "grid" else (frames * channels,)
    return build_policy(config, ObservationSpec(observation, shape, frames, action_channels=4 if temporal else 0), 4)


def _parameter_count(config: ModelConfig, observation: str) -> int:
    return sum(p.numel() for p in _build(config, observation).parameters())


@pytest.mark.parametrize("observation", ["grid", "vector"])
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_size_fields_change_the_model_exactly_when_listed(kind, observation) -> None:
    base = ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=1, mlp_ratio=1, conv_channels=8, conv_layers=1)
    reference = _parameter_count(base, observation)
    listed = KIND_INFO[kind].fields_for(observation)
    for name, value in SIZE_CHANGES.items():
        changed = _parameter_count(dataclasses.replace(base, **{name: value}), observation)
        assert (changed != reference) == (name in listed), f"{kind} on {observation}: {name}"


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_num_heads_and_dropout_listed_exactly_when_the_kind_has_attention(kind) -> None:
    has_attention = SCHEMA["model_kinds"][kind]["attention_view"] is not None
    assert ("num_heads" in KIND_INFO[kind].fields) == has_attention
    assert ("dropout" in KIND_INFO[kind].fields) == has_attention


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_unlisted_num_heads_does_not_change_outputs(kind) -> None:
    base = ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=1, conv_channels=8, conv_layers=1)
    if "num_heads" in KIND_INFO[kind].fields:
        pytest.skip("num_heads is used by this kind")
    torch.manual_seed(0)
    a = _build(base)
    torch.manual_seed(0)
    b = _build(dataclasses.replace(base, num_heads=4))
    obs = torch.rand(2, *a.observation_shape)
    assert torch.equal(a(obs).logits, b(obs).logits)


TINY_LEARNERS = {
    "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
    "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8},
}


@pytest.mark.parametrize(
    ("algorithm", "switches"),
    [(name, {}) for name in algorithm_names()] + [("dqn", {"prioritized": True})],
)
def test_each_algorithm_row_is_truthful(algorithm: str, switches: dict) -> None:
    """The schema's head is the one the policy gets, its fields are the section's, and its metrics are exactly what it logs."""
    from attentive_snake.config.experiment_io import experiment_from_tables
    from attentive_snake.factory import make_policy, make_trainer, make_vec_env
    from attentive_snake.models.heads import HEADS
    from attentive_snake.training import NullLogger

    row = SCHEMA["algorithms"][algorithm]
    assert row["section"] == algorithm and row["head"] in HEADS and row["label"] and row["description"]
    assert all(metric["label"] and metric["description"] for metric in row["metrics"])
    config = experiment_from_tables({
        "game": {"width": 4, "height": 4},
        "model": {"kind": "mlp", "d_model": 8, "num_layers": 1},
        "train": {"algorithm": algorithm, "total_env_steps": 64, "device": "cpu", "log_interval": 1},
        algorithm: TINY_LEARNERS[algorithm] | switches,
        "env": {"num_envs": 2},
    })
    env = make_vec_env(config)
    policy = make_policy(config, env)
    assert list(policy.net)[-len(HEADS[row["head"]].modules(8, 4)):] == list(HEADS[row["head"]].modules(8, 4))
    records = make_trainer(config, policy, env, NullLogger()).train()
    shared = {"iteration", "env_steps", "episodes", "mean_score", "mean_length", "mean_return", "learning_rate", "steps_per_second"}
    logged = set().union(*records) - shared
    assert logged == {metric["key"] for metric in row["metrics"] if metric["setting"] is None or switches.get(metric["setting"])}
