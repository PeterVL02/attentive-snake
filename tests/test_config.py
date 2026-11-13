import dataclasses
import tomllib
from pathlib import Path

import pytest

from attentive_snake.config import EnvConfig, ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.toml_writer import format_tables, format_value
from attentive_snake.game.config import GameConfig
from attentive_snake.training import TrainConfig

DEFAULT_TOML = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


@dataclasses.dataclass(frozen=True)
class Sample:
    count: int = 1
    rate: float = 0.5
    flag: bool = False
    name: str = "x"
    folder: Path | None = None
    limit: int | None = None


# --------------------------------------------------------------------------- writer


@pytest.mark.parametrize(
    "value",
    [0, -7, 10**12, 0.5, 1e-5, 2.0, -3.25e10, True, False, "", "plain", 'quote " and \\ back', "tab\tnew\nline", "ünï"],
)
def test_format_value_round_trips_through_tomllib(value) -> None:
    parsed = tomllib.loads(f"v = {format_value(value)}")["v"]
    assert parsed == value and type(parsed) is type(value)


def test_format_value_rejects_non_scalars() -> None:
    for value in (None, [1], {"a": 1}, Path("x")):
        with pytest.raises(TypeError):
            format_value(value)


def test_format_tables_is_parseable() -> None:
    text = format_tables({"a": {"x": 1, "y": "s"}, "b": {}})
    assert tomllib.loads(text) == {"a": {"x": 1, "y": "s"}, "b": {}}


# --------------------------------------------------------------------------- dataclass <-> table


def test_dataclass_table_round_trip_with_path_and_none() -> None:
    sample = Sample(count=3, rate=0.25, flag=True, name="n", folder=Path("runs/a"))
    table = dataclass_to_table(sample)
    assert table == {"count": 3, "rate": 0.25, "flag": True, "name": "n", "folder": "runs/a"}  # limit omitted
    assert dataclass_from_table(Sample, tomllib.loads(format_tables({"s": table}))["s"]) == sample


def test_absent_keys_take_defaults_and_ints_widen_to_float() -> None:
    assert dataclass_from_table(Sample, {"rate": 2}) == Sample(rate=2.0)
    assert type(dataclass_from_table(Sample, {"rate": 2}).rate) is float


def test_unknown_key_names_key_and_valid_keys() -> None:
    with pytest.raises(ValueError, match=r"\[sec\] unknown key 'cout'.*valid keys: count, rate, flag"):
        dataclass_from_table(Sample, {"cout": 1}, section="sec")


@pytest.mark.parametrize("key, value", [("count", 1.5), ("count", True), ("flag", 1), ("name", 3), ("rate", "x")])
def test_wrong_types_are_rejected(key, value) -> None:
    with pytest.raises(TypeError, match=key):
        dataclass_from_table(Sample, {key: value})


def test_none_in_field_with_non_none_default_cannot_be_written() -> None:
    @dataclasses.dataclass
    class Bad:
        value: int | None = 3

    with pytest.raises(ValueError, match="round trip"):
        dataclass_to_table(Bad(value=None))


# --------------------------------------------------------------------------- experiment


def test_experiment_save_load_round_trip(tmp_path) -> None:
    config = ExperimentConfig(
        game=GameConfig(width=6, height=5, max_steps_without_food=40),
        train=TrainConfig(total_env_steps=1000, target_kl=0.03, checkpoint_dir=Path("runs/x")),
        env=EnvConfig(num_envs=3),
    )
    path = tmp_path / "nested" / "config.toml"
    save_experiment(config, path)
    assert load_experiment(path) == config


def test_missing_sections_are_defaults_and_unknown_sections_fail(tmp_path) -> None:
    assert experiment_from_tables({}) == ExperimentConfig()
    assert experiment_from_tables({"env": {"num_envs": 2}}) == ExperimentConfig(env=EnvConfig(num_envs=2))
    with pytest.raises(ValueError, match=r"unknown section \[games\]"):
        experiment_from_tables({"games": {}})
    path = tmp_path / "bad.toml"
    path.write_text("[model]\nd_modl = 3\n")
    with pytest.raises(ValueError, match=r"bad.toml: \[model\] unknown key 'd_modl'"):
        load_experiment(path)


def test_dataclass_validation_still_runs() -> None:
    with pytest.raises(ValueError):
        experiment_from_tables({"env": {"num_envs": 0}})


def test_default_toml_loads_and_lists_every_field() -> None:
    config = load_experiment(DEFAULT_TOML)
    assert (config.game.width, config.game.height, config.game.max_steps_without_food) == (8, 8, 128)
    tables = tomllib.loads(DEFAULT_TOML.read_text())
    for section in dataclasses.fields(ExperimentConfig):
        listed = tables[section.name]
        for field in dataclasses.fields(section.type):
            # Fields defaulting to None can only be listed as comments.
            assert field.name in listed or field.default is None, f"[{section.name}] {field.name} missing"


def test_default_toml_model_and_env_history_keys() -> None:
    config = load_experiment(DEFAULT_TOML)
    assert config.model.kind == "transformer"
    assert (config.env.history_length, config.env.history_actions) == (1, False)
