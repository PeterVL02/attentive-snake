import dataclasses
import tomllib
from pathlib import Path

import pytest

from attentive_snake.config import EnvConfig, ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.dataclass_table import dataclass_from_table, dataclass_to_table
from attentive_snake.config.experiment import SECTIONS, section_types
from attentive_snake.config.experiment_io import experiment_from_tables, experiment_to_tables
from attentive_snake.config.layout import default_config_path, rewards_dir
from attentive_snake.config.reward_schemes import load_scheme
from attentive_snake.games.registry import game_names, get_game
from attentive_snake.config.toml_writer import format_tables, format_value
from attentive_snake.games.snake.reward import SnakeRewardConfig
from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.training import TrainConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_TOML = CONFIGS / "snake" / "default.toml"


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
        game=SnakeConfig(width=6, height=5, max_steps_without_food=40),
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


def test_default_toml_loads() -> None:
    config = load_experiment(DEFAULT_TOML)
    assert config.game_name == "snake"
    assert (config.game.width, config.game.height, config.game.max_steps_without_food) == (8, 8, 128)
    assert config.reward == SnakeRewardConfig()


@pytest.mark.parametrize("game", game_names())
def test_every_game_has_a_default_config_listing_every_field_and_a_default_scheme(game) -> None:
    """configs/<game>/default.toml names the game and lists every field but the rewards;
    configs/<game>/rewards/default.toml is the single source of the default reward values."""
    path = default_config_path(CONFIGS, game)
    config = load_experiment(path)
    assert config.game_name == game
    tables = tomllib.loads(path.read_text())
    assert tables["game"]["name"] == game
    assert "reward" not in tables  # the default reward values live only in the default scheme
    types = section_types(game)
    for section in SECTIONS:
        if section == "reward":
            continue
        for field in dataclasses.fields(types[section]):
            # Fields defaulting to None can only be listed as comments.
            assert field.name in tables[section] or field.default is None, f"[{section}] {field.name} missing"
    reward_type = get_game(game).reward_type
    listed = tomllib.loads((rewards_dir(CONFIGS, game) / "default.toml").read_text())
    assert set(listed) == {field.name for field in dataclasses.fields(reward_type)}
    assert load_scheme(rewards_dir(CONFIGS, game), "default", reward_type) == reward_type() == config.reward


def test_game_name_selects_the_game_and_defaults_to_snake(tmp_path) -> None:
    assert ExperimentConfig().game_name == "snake" and ExperimentConfig().game == SnakeConfig()
    legacy = tmp_path / "legacy.toml"
    legacy.write_text("[game]\nwidth = 6\n")  # a config from before there were several games
    assert load_experiment(legacy) == ExperimentConfig(game=SnakeConfig(width=6))
    assert experiment_to_tables(ExperimentConfig())["game"]["name"] == "snake"
    with pytest.raises(ValueError, match=r"\[game\] name: unknown game 'chess'"):
        experiment_from_tables({"game": {"name": "chess"}})
    with pytest.raises(TypeError, match="name must be str"):
        experiment_from_tables({"game": {"name": 3}})
    with pytest.raises(TypeError, match="SnakeConfig"):
        ExperimentConfig(game=SnakeRewardConfig())


def test_default_toml_model_and_env_history_keys() -> None:
    config = load_experiment(DEFAULT_TOML)
    assert config.model.kind == "transformer"
    assert (config.env.history_length, config.env.history_actions) == (1, False)
