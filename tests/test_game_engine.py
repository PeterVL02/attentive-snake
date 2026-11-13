from collections import deque

import pytest

from attentive_snake.game.config import GameConfig
from attentive_snake.game.engine import SnakeGame
from attentive_snake.game.types import Action, StepEvent


def make_game(config: GameConfig, snake, direction=Action.RIGHT, food=None, seed=0) -> SnakeGame:
    """Build a game in a hand-crafted state (pokes private attributes)."""
    game = SnakeGame(config, seed=seed)
    game._snake = deque(snake)
    game._occupied = set(snake)
    game._direction = direction
    game._food = food
    return game


def test_reset_layout():
    game = SnakeGame(GameConfig(width=10, height=8, initial_length=3), seed=1)
    state = game.state
    ys = {y for _, y in state.snake}
    xs = [x for x, _ in state.snake]
    assert len(ys) == 1
    assert xs == sorted(xs, reverse=True) and xs[0] - xs[-1] == 2
    assert state.direction == Action.RIGHT
    assert state.food is not None and state.food not in state.snake
    assert (state.score, state.steps, state.steps_since_food, state.done) == (0, 0, 0, False)


def test_movement():
    game = SnakeGame(GameConfig(width=10, height=10), seed=1)
    before = game.state.snake
    result = game.step(Action.RIGHT)
    assert result.event in (StepEvent.MOVED, StepEvent.ATE_FOOD)
    assert result.state.head == (before[0][0] + 1, before[0][1])
    assert result.state.steps == 1
    game.step(Action.DOWN)
    assert game.state.direction == Action.DOWN


def test_reverse_action_ignored():
    game = SnakeGame(GameConfig(width=10, height=10), seed=1)
    head = game.state.head
    game.step(Action.LEFT)
    assert game.state.direction == Action.RIGHT
    assert game.state.head == (head[0] + 1, head[1])


def test_wall_death_keeps_snake():
    game = make_game(GameConfig(width=5, height=5), [(1, 0), (2, 0), (3, 0)], Action.LEFT, food=(4, 4))
    result = game.step(Action.UP)
    assert result.event is StepEvent.HIT_WALL
    assert result.state.done
    assert result.state.snake == ((1, 0), (2, 0), (3, 0))


def test_wall_death_all_sides():
    config = GameConfig(width=3, height=3, initial_length=1)
    for start, action in [((0, 1), Action.LEFT), ((2, 1), Action.RIGHT), ((1, 2), Action.DOWN)]:
        game = make_game(config, [start], action, food=(1, 1))
        assert game.step(action).event is StepEvent.HIT_WALL


def test_self_collision():
    # Head (2,2) moves UP into (2,1), a body cell (not the tail).
    snake = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1)]
    game = make_game(GameConfig(width=6, height=6), snake, Action.UP, food=(5, 5))
    result = game.step(Action.UP)
    assert result.event is StepEvent.HIT_SELF
    assert result.state.done
    assert result.state.snake == tuple(snake)


def test_moving_into_vacating_tail_is_legal():
    snake = [(2, 2), (2, 3), (3, 3), (3, 2)]  # 2x2 loop; head's right neighbour is the tail
    game = make_game(GameConfig(width=6, height=6), snake, Action.UP, food=(5, 5))
    result = game.step(Action.RIGHT)
    assert result.event is StepEvent.MOVED
    assert result.state.snake == ((3, 2), (2, 2), (2, 3), (3, 3))


def test_moving_into_tail_while_growing_is_fatal():
    snake = [(2, 2), (2, 3), (3, 3), (3, 2)]
    game = make_game(GameConfig(width=6, height=6), snake, Action.UP, food=(3, 2))
    assert game.step(Action.RIGHT).event is StepEvent.HIT_SELF


def test_eating_grows_and_scores():
    game = make_game(GameConfig(width=6, height=6), [(2, 2), (1, 2)], Action.RIGHT, food=(3, 2))
    result = game.step(Action.RIGHT)
    assert result.event is StepEvent.ATE_FOOD
    assert result.state.snake == ((3, 2), (2, 2), (1, 2))
    assert result.state.score == 1
    assert result.state.steps_since_food == 0
    assert result.state.food is not None and result.state.food not in result.state.snake


def test_starvation():
    config = GameConfig(width=10, height=10, max_steps_without_food=3)
    game = make_game(config, [(2, 5), (1, 5)], Action.RIGHT, food=(9, 0))
    assert game.step(Action.RIGHT).event is StepEvent.MOVED
    assert game.step(Action.RIGHT).event is StepEvent.MOVED
    result = game.step(Action.RIGHT)
    assert result.event is StepEvent.STARVED
    assert result.state.done
    assert result.state.steps_since_food == 3


def test_eating_resets_starvation_counter():
    config = GameConfig(width=10, height=10, max_steps_without_food=2)
    game = make_game(config, [(2, 5), (1, 5)], Action.RIGHT, food=(4, 5))
    game.step(Action.RIGHT)
    assert game.step(Action.RIGHT).event is StepEvent.ATE_FOOD
    assert game.state.steps_since_food == 0


def test_win_on_tiny_grid():
    # 3x2 grid: snake of 5 cells, food in the last free cell.
    snake = [(1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
    game = make_game(GameConfig(width=3, height=2, initial_length=3), snake, Action.LEFT, food=(0, 0))
    result = game.step(Action.LEFT)
    assert result.event is StepEvent.WON
    assert result.state.done
    assert result.state.food is None
    assert len(result.state.snake) == 6
    assert result.state.score == 1


def test_determinism_with_seed():
    def run(seed):
        game = SnakeGame(GameConfig(width=8, height=8), seed=seed)
        actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP] * 5
        trace = [game.state]
        for action in actions:
            if game.state.done:
                break
            trace.append(game.step(action).state)
        return trace

    assert run(42) == run(42)


def test_reset_with_seed_reproduces_food():
    game = SnakeGame(GameConfig(width=12, height=12))
    first = game.reset(seed=7)
    game.step(Action.UP)
    assert game.reset(seed=7) == first


def test_non_square_grid():
    config = GameConfig(width=7, height=3, initial_length=3)
    game = SnakeGame(config, seed=3)
    state = game.state
    assert (state.width, state.height) == (7, 3)
    assert all(0 <= x < 7 and 0 <= y < 3 for x, y in state.snake)
    game = make_game(config, [(5, 1), (4, 1), (3, 1)], Action.RIGHT, food=(0, 0))
    game.step(Action.RIGHT)
    assert game.step(Action.RIGHT).event is StepEvent.HIT_WALL  # x == 7 is outside


def test_step_after_done_raises():
    game = make_game(GameConfig(width=3, height=3, initial_length=1), [(0, 0)], Action.UP, food=(2, 2))
    game.step(Action.UP)
    with pytest.raises(RuntimeError):
        game.step(Action.UP)


def test_reset_after_done_allows_play():
    game = make_game(GameConfig(width=3, height=3, initial_length=1), [(0, 0)], Action.UP, food=(2, 2))
    game.step(Action.UP)
    assert not game.reset().done
    game.step(Action.RIGHT)
