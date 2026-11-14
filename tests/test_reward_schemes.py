import dataclasses
import tomllib
from pathlib import Path

import pytest

from attentive_snake.config.reward_schemes import delete_scheme, list_schemes, load_scheme, save_scheme
from attentive_snake.games.snake.reward_table import reward_table
from attentive_snake.games.snake.reward import SnakeRewardConfig

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "snake" / "rewards"


def test_save_load_list_delete_round_trip(tmp_path) -> None:
    assert list_schemes(tmp_path / "missing") == []
    config = SnakeRewardConfig(food=2.5, closer_to_food=0.1, step=-0.5)
    path = save_scheme(tmp_path / "rewards", "mine", config)
    assert path == tmp_path / "rewards" / "mine.toml"
    assert "[" not in path.read_text()  # flat, no section header
    assert load_scheme(tmp_path / "rewards", "mine", SnakeRewardConfig) == config
    save_scheme(tmp_path / "rewards", "another", SnakeRewardConfig())
    assert list_schemes(tmp_path / "rewards") == ["another", "mine"]
    delete_scheme(tmp_path / "rewards", "mine")
    assert list_schemes(tmp_path / "rewards") == ["another"]


@pytest.mark.parametrize("name", ["../x", ".hidden", "a/b", ""])
def test_names_are_validated(tmp_path, name) -> None:
    for call in (
        lambda: save_scheme(tmp_path, name, SnakeRewardConfig()),
        lambda: load_scheme(tmp_path, name, SnakeRewardConfig),
        lambda: delete_scheme(tmp_path, name),
    ):
        with pytest.raises(ValueError):
            call()


def test_unknown_key_in_a_scheme_file_is_an_error(tmp_path) -> None:
    (tmp_path / "bad.toml").write_text("foood = 1.0\n")
    with pytest.raises(ValueError, match="bad.toml"):
        load_scheme(tmp_path, "bad", SnakeRewardConfig)


def test_shipped_starter_schemes() -> None:
    assert {"big_win", "default", "distance_shaping"} <= set(list_schemes(SHIPPED))  # user-added schemes are fine
    assert load_scheme(SHIPPED, "default", SnakeRewardConfig) == SnakeRewardConfig()
    big = load_scheme(SHIPPED, "big_win", SnakeRewardConfig)
    assert big.win > SnakeRewardConfig().win and big.food_growth > 0
    assert big.closer_to_food == 0 and big.farther_from_food == 0
    shaping = load_scheme(SHIPPED, "distance_shaping", SnakeRewardConfig)
    assert shaping.closer_to_food > 0 and shaping.farther_from_food < 0


def _rewards(config: SnakeRewardConfig, **kwargs) -> dict[str, float]:
    return {row["situation"]: row["reward"] for row in reward_table(config, **kwargs)}


def test_reward_table_rows_cover_every_situation_and_are_consistent() -> None:
    rows = reward_table(SnakeRewardConfig())
    assert len(rows) == 9
    for row in rows:
        assert set(row) == {"situation", "reward", "breakdown"}
        assert row["reward"] == pytest.approx(sum(row["breakdown"].values()))


def test_reward_table_default_values() -> None:
    rewards = _rewards(SnakeRewardConfig())
    assert rewards["Eat food at score 0"] == pytest.approx(0.99)
    assert rewards["Eat food at score 5"] == pytest.approx(0.99)
    assert rewards["Hit a wall"] == -1.0 and rewards["Hit itself"] == -1.0
    assert rewards["Starve (too long without food)"] == -1.0
    assert rewards["Win (fill the board)"] == 1.0
    assert rewards["Move closer to the food"] == rewards["Move farther from the food"] == pytest.approx(-0.01)


def test_reward_table_reflects_growth_and_shaping() -> None:
    growth = _rewards(SnakeRewardConfig(food=1.0, food_growth=0.5, step=0.0), example_score=4)
    assert growth["Eat food at score 0"] == pytest.approx(1.0)
    assert growth["Eat food at score 4"] == pytest.approx(3.0)
    shaped = _rewards(SnakeRewardConfig(closer_to_food=0.1, farther_from_food=-0.2, step=0.0))
    assert shaped["Move closer to the food"] == pytest.approx(0.1)
    assert shaped["Move farther from the food"] == pytest.approx(-0.2)
    assert shaped["Plain move (ignoring distance to food)"] == 0.0


def test_default_scheme_is_the_single_source_of_the_default_reward_values() -> None:
    """configs/snake/rewards/default.toml lists every SnakeRewardConfig field with exactly the code defaults."""
    listed = tomllib.loads((SHIPPED / "default.toml").read_text())
    assert set(listed) == {field.name for field in dataclasses.fields(SnakeRewardConfig)}
    assert load_scheme(SHIPPED, "default", SnakeRewardConfig) == SnakeRewardConfig()
