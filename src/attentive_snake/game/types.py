"""Plain data types shared by the game engine and everything that consumes it.

This module is the contract between the game and the outside world (renderer,
RL environment, human play). It contains no game logic.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum

# (x, y) grid coordinates. x grows to the right, y grows downwards, (0, 0) is top-left.
Position = tuple[int, int]


class Action(IntEnum):
    """Absolute movement direction. Also used as the snake's current heading.

    An action pointing opposite to the current heading is ignored by the engine
    (the snake keeps going straight), as in the classic game.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Position:
        return _DELTAS[self]

    @property
    def opposite(self) -> "Action":
        return Action((self + 2) % 4)


_DELTAS: dict[Action, Position] = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}


class StepEvent(Enum):
    """What happened during one game step."""

    MOVED = "moved"
    ATE_FOOD = "ate_food"
    HIT_WALL = "hit_wall"
    HIT_SELF = "hit_self"
    WON = "won"  # the snake fills the whole grid
    STARVED = "starved"  # GameConfig.max_steps_without_food exceeded

    @property
    def is_terminal(self) -> bool:
        """The game ended because of the game rules (death or win)."""
        return self in (StepEvent.HIT_WALL, StepEvent.HIT_SELF, StepEvent.WON)

    @property
    def is_truncation(self) -> bool:
        """The game was cut short by a limit rather than by the rules."""
        return self is StepEvent.STARVED

    @property
    def ends_game(self) -> bool:
        return self.is_terminal or self.is_truncation


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a game.

    Attributes:
        width, height: Grid size.
        snake: Body cells, head first, tail last.
        direction: Current heading of the snake.
        food: Food cell, or ``None`` when the grid is full (game won).
        score: Number of food items eaten.
        steps: Steps taken since the game started.
        steps_since_food: Steps taken since food was last eaten (or since start).
        done: Whether the game is over; a finished game must be reset.
    """

    width: int
    height: int
    snake: tuple[Position, ...]
    direction: Action
    food: Position | None
    score: int
    steps: int
    steps_since_food: int
    done: bool

    @property
    def head(self) -> Position:
        return self.snake[0]


@dataclass(frozen=True)
class StepResult:
    """Outcome of ``SnakeGame.step``: the new state and what happened."""

    state: GameState
    event: StepEvent
