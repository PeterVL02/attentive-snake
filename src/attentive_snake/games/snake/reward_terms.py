"""Single-purpose reward terms. Each is pure, stateless and returns 0 when it does not apply."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from attentive_snake.games.snake.types import GameState, Position, StepEvent, StepResult


class RewardTerm(Protocol):
    """A reward component: maps one transition to a float. Must be pure and stateless."""

    @property
    def name(self) -> str: ...

    def __call__(self, previous: GameState, result: StepResult) -> float: ...


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class FoodReward:
    """On ATE_FOOD: ``food + growth * score_before_eating``.

    The first food (score 0 before) pays ``food``, the k-th (1-based) pays
    ``food + growth * (k - 1)``.
    """

    food: float
    growth: float = 0.0
    name: ClassVar[str] = "food"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        if result.event is not StepEvent.ATE_FOOD:
            return 0.0
        return self.food + self.growth * previous.score


@dataclass(frozen=True)
class DeathPenalty:
    """``value`` on HIT_WALL or HIT_SELF."""

    value: float
    name: ClassVar[str] = "death"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        died = result.event in (StepEvent.HIT_WALL, StepEvent.HIT_SELF)
        return self.value if died else 0.0


@dataclass(frozen=True)
class WinReward:
    """``value`` on WON (the snake fills the grid)."""

    value: float
    name: ClassVar[str] = "win"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        return self.value if result.event is StepEvent.WON else 0.0


@dataclass(frozen=True)
class StarvationPenalty:
    """``value`` on STARVED (step limit without food reached)."""

    value: float
    name: ClassVar[str] = "starvation"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        return self.value if result.event is StepEvent.STARVED else 0.0


@dataclass(frozen=True)
class StepCost:
    """``value`` on every step that does not end the game (MOVED and ATE_FOOD)."""

    value: float
    name: ClassVar[str] = "step_cost"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        return 0.0 if result.event.ends_game else self.value


@dataclass(frozen=True)
class FoodDistanceShaping:
    """On a plain MOVED step: ``closer`` if the head's Manhattan distance to the
    food decreased, ``farther`` if it increased, else 0."""

    closer: float = 0.0
    farther: float = 0.0
    name: ClassVar[str] = "food_distance"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        new = result.state
        if result.event is not StepEvent.MOVED or previous.food is None or new.food is None:
            return 0.0
        before = _manhattan(previous.head, previous.food)
        after = _manhattan(new.head, new.food)
        if after < before:
            return self.closer
        if after > before:
            return self.farther
        return 0.0
