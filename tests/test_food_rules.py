"""The two apple rules: apples near the centre (food_distribution) and several at once (food_count)."""

import random
from collections import Counter, deque

import numpy as np
import pytest

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.engine import SnakeGame
from attentive_snake.games.snake.food_placement import FoodPlacer
from attentive_snake.games.snake.observation import FOOD, ObservationEncoder
from attentive_snake.games.snake.reward_terms import FoodDistanceShaping
from attentive_snake.games.snake.types import Action, GameState, StepEvent, StepResult
from attentive_snake.games.snake.view import board_view
from attentive_snake.views.board_window import BoardWindow


def crafted(config: SnakeConfig, snake, foods, direction=Action.RIGHT, seed=0) -> SnakeGame:
    game = SnakeGame(config, seed=seed)
    game._snake = deque(snake)
    game._occupied = set(snake)
    game._direction = direction
    game._foods = list(foods)
    return game


# --------------------------------------------------------------------------- config


def test_defaults_and_validation() -> None:
    config = SnakeConfig()
    assert (config.food_distribution, config.food_spread, config.food_count) == ("uniform", 2.0, 1)
    for bad in ({"food_distribution": "corner"}, {"food_spread": 0.0}, {"food_spread": -1.0}, {"food_count": 0}):
        with pytest.raises(ValueError):
            SnakeConfig(**bad)


# --------------------------------------------------------------------------- placement


def test_uniform_placement_draws_like_rng_choice() -> None:
    placer = FoodPlacer(SnakeConfig(width=5, height=5))
    free = [(x, y) for y in range(5) for x in range(5)]
    assert [placer.choose(free, random.Random(s)) for s in range(20)] == [random.Random(s).choice(free) for s in range(20)]


def test_centre_weighting_favours_the_middle_and_never_leaves_the_free_cells() -> None:
    config = SnakeConfig(width=9, height=9, food_distribution="center", food_spread=1.5)
    placer = FoodPlacer(config)
    rng = random.Random(0)
    free = [(x, y) for y in range(9) for x in range(9) if (x, y) not in {(4, 4), (3, 4), (5, 4)}]
    drawn = Counter(placer.choose(free, rng) for _ in range(4000))
    assert set(drawn) <= set(free)
    near = sum(n for (x, y), n in drawn.items() if (x - 4) ** 2 + (y - 4) ** 2 <= 4)
    corners = sum(n for (x, y), n in drawn.items() if abs(x - 4) >= 3 and abs(y - 4) >= 3)
    assert near > 0.4 * 4000  # 10 of the 78 free cells are that near: uniform would give about 13 % and corners < 0.02 * 4000


def test_centre_weights_follow_the_formula() -> None:
    placer = FoodPlacer(SnakeConfig(width=4, height=4, food_distribution="center", food_spread=1.0))
    weights = placer._weights
    assert weights[(1, 1)] == pytest.approx(np.exp(-0.25))  # d^2 = 0.5, centre (1.5, 1.5)
    assert weights[(0, 0)] == pytest.approx(np.exp(-2.25))  # d^2 = 4.5


def test_centre_games_never_put_an_apple_on_the_snake() -> None:
    game = SnakeGame(SnakeConfig(width=6, height=6, food_distribution="center", food_count=3), seed=1)
    chooser = random.Random(1)
    for _ in range(300):
        result = game.step(Action(chooser.randrange(4)))
        state = result.state
        assert not set(state.foods) & set(state.snake) or state.done
        if state.done:
            game.reset()


# --------------------------------------------------------------------------- several apples


def test_reset_places_food_count_distinct_apples() -> None:
    state = SnakeGame(SnakeConfig(width=6, height=6, food_count=3), seed=2).state
    assert len(state.foods) == 3 and len(set(state.foods)) == 3 and not set(state.foods) & set(state.snake)


def test_fewer_apples_when_there_is_no_room() -> None:
    state = SnakeGame(SnakeConfig(width=3, height=2, initial_length=3, food_count=9), seed=0).state
    assert len(state.foods) == 3 and not set(state.foods) & set(state.snake)


def test_eating_one_apple_spawns_exactly_one_new_one() -> None:
    config = SnakeConfig(width=6, height=6, initial_length=2, food_count=3)
    game = crafted(config, [(2, 2), (1, 2)], [(3, 2), (5, 5), (0, 5)])
    result = game.step(Action.RIGHT)
    assert result.event is StepEvent.ATE_FOOD and result.state.score == 1
    foods = result.state.foods
    assert len(foods) == 3 and len(set(foods)) == 3 and (3, 2) not in foods
    assert {(5, 5), (0, 5)} <= set(foods) and not set(foods) & set(result.state.snake)


def test_no_spawn_when_no_cell_is_left_and_the_game_goes_on() -> None:
    # 3x2: snake of 4 cells, two apples fill the board; eating one leaves the other.
    snake = [(1, 0), (2, 0), (2, 1), (1, 1)]
    config = SnakeConfig(width=3, height=2, initial_length=3, food_count=2)
    game = crafted(config, snake, [(0, 0), (0, 1)], Action.LEFT)
    result = game.step(Action.LEFT)
    assert result.event is StepEvent.ATE_FOOD and result.state.foods == ((0, 1),)
    assert not result.state.done


def test_filling_the_board_with_several_apples_wins() -> None:
    snake = [(1, 0), (2, 0), (2, 1), (1, 1)]
    config = SnakeConfig(width=3, height=2, initial_length=3, food_count=2)
    game = crafted(config, snake, [(0, 0), (0, 1)], Action.LEFT)
    game.step(Action.LEFT)
    result = game.step(Action.DOWN)
    assert result.event is StepEvent.WON and result.state.done and result.state.foods == ()
    assert len(result.state.snake) == 6


def test_single_apple_count_is_one_throughout() -> None:
    game = SnakeGame(SnakeConfig(width=6, height=6), seed=4)
    chooser = random.Random(4)
    for _ in range(200):
        result = game.step(Action(chooser.randrange(4)))
        assert len(result.state.foods) == 1 or result.state.done
        if result.state.done:
            game.reset()


# --------------------------------------------------------------------------- observation, view, rewards


def state_with(foods, head=(2, 2)) -> GameState:
    return GameState(6, 5, (head, (head[0] - 1, head[1])), Action.RIGHT, tuple(foods), 0, 0, 0, False)


def test_observation_marks_every_apple_and_keeps_its_shape() -> None:
    encoder = ObservationEncoder()
    obs = encoder.encode(state_with([(4, 1), (0, 4), (5, 0)]))
    assert obs.shape == (3, 5, 6) == encoder.shape(6, 5)
    assert obs[FOOD].sum() == 3 and obs[FOOD, 1, 4] == obs[FOOD, 4, 0] == obs[FOOD, 0, 5] == 1.0
    assert encoder.encode(state_with([]))[FOOD].sum() == 0


def test_view_payload_lists_every_apple() -> None:
    view = board_view(state_with([(4, 1), (0, 4)]))
    assert view["foods"] == [[4, 1], [0, 4]] and view["snake"] == [[2, 2], [1, 2]]
    assert board_view(state_with([]))["foods"] == []


def test_board_window_draws_every_apple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    window = BoardWindow(6, 5, cell_size=10)
    drawn = []
    monkeypatch.setattr(window, "_draw_cell", lambda screen, cell, color: drawn.append((tuple(cell), color)))
    try:
        window.render(board_view(state_with([(4, 1), (0, 4)])))
    finally:
        window.close()
    reds = [cell for cell, color in drawn if color == window._config.food]
    assert reds == [(4, 1), (0, 4)]


def test_distance_shaping_uses_the_nearest_apple() -> None:
    term = FoodDistanceShaping(closer=1.0, farther=-1.0)
    previous = state_with([(5, 2), (0, 4)])  # head (2, 2): nearest is 3 away
    towards = StepResult(state_with([(5, 2), (0, 4)], head=(3, 2)), StepEvent.MOVED)
    away = StepResult(state_with([(5, 2), (0, 4)], head=(2, 1)), StepEvent.MOVED)
    assert term(previous, towards) == 1.0 and term(previous, away) == -1.0
    # the nearest apple stays 2 away (the other one moves from 2 to 3): no change
    assert term(state_with([(0, 2), (5, 2)], head=(2, 2)), StepResult(state_with([(0, 2), (5, 2)], head=(3, 2)), StepEvent.MOVED)) == 0.0
    assert term(previous, StepResult(state_with([], head=(3, 2)), StepEvent.MOVED)) == 0.0
