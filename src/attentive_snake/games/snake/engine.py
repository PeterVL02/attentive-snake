"""Pure snake game logic: no rendering, no rewards, no numpy/torch/pygame."""

import random
from collections import deque

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.types import Action, GameState, Position, StepEvent, StepResult


class SnakeGame:
    """Classic snake on a rectangular grid with walls.

    The snake body is a deque (head first) mirrored by a set of occupied cells,
    so a step costs O(1) apart from building the immutable state snapshot.
    """

    def __init__(self, config: SnakeConfig, seed: int | None = None) -> None:
        self._config = config
        self._rng = random.Random(seed)
        self._snake: deque[Position] = deque()
        self._occupied: set[Position] = set()
        self._direction = Action.RIGHT
        self._food: Position | None = None
        self._score = 0
        self._steps = 0
        self._steps_since_food = 0
        self._done = False
        self.reset()

    @property
    def state(self) -> GameState:
        return GameState(
            width=self._config.width,
            height=self._config.height,
            snake=tuple(self._snake),
            direction=self._direction,
            food=self._food,
            score=self._score,
            steps=self._steps,
            steps_since_food=self._steps_since_food,
            done=self._done,
        )

    def reset(self, seed: int | None = None) -> GameState:
        """Start a new game, optionally reseeding the RNG first."""
        if seed is not None:
            self._rng.seed(seed)
        self._place_initial_snake()
        self._direction = Action.RIGHT
        self._score = 0
        self._steps = 0
        self._steps_since_food = 0
        self._done = False
        self._food = self._spawn_food()
        return self.state

    def step(self, action: Action) -> StepResult:
        """Advance the game by one move."""
        if self._done:
            raise RuntimeError("Game is over; call reset() first.")
        direction = self._effective_direction(action)
        self._steps += 1
        self._direction = direction
        new_head = _move(self._snake[0], direction)

        collision = self._collision_event(new_head)
        if collision is not None:
            return self._finish(collision)

        ate = new_head == self._food
        self._advance_snake(new_head, grow=ate)
        if ate:
            return self._after_eating()
        self._steps_since_food += 1
        if self._is_starved():
            return self._finish(StepEvent.STARVED)
        return StepResult(self.state, StepEvent.MOVED)

    def _effective_direction(self, action: Action) -> Action:
        return self._direction if action == self._direction.opposite else action

    def _collision_event(self, new_head: Position) -> StepEvent | None:
        if not self._in_bounds(new_head):
            return StepEvent.HIT_WALL
        grows = new_head == self._food
        # The tail cell is vacated this step unless the snake grows.
        tail_vacates = not grows and new_head == self._snake[-1]
        if new_head in self._occupied and not tail_vacates:
            return StepEvent.HIT_SELF
        return None

    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self._config.width and 0 <= pos[1] < self._config.height

    def _advance_snake(self, new_head: Position, grow: bool) -> None:
        if not grow:
            self._occupied.discard(self._snake.pop())
        self._snake.appendleft(new_head)
        self._occupied.add(new_head)

    def _after_eating(self) -> StepResult:
        self._score += 1
        self._steps_since_food = 0
        self._food = self._spawn_food()
        if self._food is None:
            return self._finish(StepEvent.WON)
        return StepResult(self.state, StepEvent.ATE_FOOD)

    def _is_starved(self) -> bool:
        limit = self._config.max_steps_without_food
        return limit is not None and self._steps_since_food >= limit

    def _finish(self, event: StepEvent) -> StepResult:
        self._done = True
        return StepResult(self.state, event)

    def _place_initial_snake(self) -> None:
        length = self._config.initial_length
        y = self._config.height // 2
        head_x = (self._config.width + length) // 2 - 1
        cells = [(head_x - i, y) for i in range(length)]
        self._snake = deque(cells)
        self._occupied = set(cells)

    def _spawn_food(self) -> Position | None:
        """Random free cell, or None when the snake fills the grid. O(grid)."""
        free = [
            (x, y)
            for y in range(self._config.height)
            for x in range(self._config.width)
            if (x, y) not in self._occupied
        ]
        return self._rng.choice(free) if free else None


def _move(pos: Position, direction: Action) -> Position:
    dx, dy = direction.delta
    return pos[0] + dx, pos[1] + dy
