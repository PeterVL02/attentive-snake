import dataclasses
import json

import pytest
import torch

from attentive_snake.config import ExperimentConfig
from attentive_snake.config.schema import experiment_schema
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy

SCHEMA = experiment_schema()


def test_sections_in_order_and_cover_every_dataclass_field() -> None:
    assert list(SCHEMA["sections"]) == ["game", "reward", "model", "train", "env"]
    experiment = ExperimentConfig()
    for section, rows in SCHEMA["sections"].items():
        fields = [f.name for f in dataclasses.fields(getattr(experiment, section))]
        assert [row["name"] for row in rows] == fields


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
    assert rows["train", "target_kl"]["type"] == "float"
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
        assert set(info) == {"description", "fields", "uses_history", "recommended_env", "attention_view"}
        assert info["description"].strip()
        assert set(info["fields"]) <= model_fields
        temporal = info["uses_history"]
        assert info["recommended_env"] == {"history_length": 4 if temporal else 1, "history_actions": temporal}
        assert info["attention_view"] in ("grid", "frames", "weights", None)
    assert {k for k, i in SCHEMA["model_kinds"].items() if i["uses_history"]} == {"temporal_transformer", "frame_attention"}
    assert SCHEMA["model_kinds"]["cnn"]["attention_view"] is None


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


def _build(config: ModelConfig):
    """A policy for a 5x5 board; temporal kinds get 2 frames with action planes (7 channels per frame)."""
    if KIND_INFO[config.kind].uses_history:
        return build_policy(config, (14, 5, 5), 4, 2)
    return build_policy(config, (3, 5, 5), 4, 1)


def _parameter_count(config: ModelConfig) -> int:
    return sum(p.numel() for p in _build(config).parameters())


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_size_fields_change_the_model_exactly_when_listed(kind) -> None:
    base = ModelConfig(kind=kind, d_model=16, num_heads=2, num_layers=1, mlp_ratio=1, conv_channels=8, conv_layers=1)
    reference = _parameter_count(base)
    for name, value in SIZE_CHANGES.items():
        changed = _parameter_count(dataclasses.replace(base, **{name: value}))
        assert (changed != reference) == (name in KIND_INFO[kind].fields), f"{kind}: {name}"


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
