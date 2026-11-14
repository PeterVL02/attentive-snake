"""What a reward scheme actually pays in each kind of step.

Each situation is a tiny hand-built (previous state, step result) pair on a 5x5 board,
scored by the pipeline that ``build_reward`` makes from the config.
"""

from attentive_snake.games.snake.reward import RewardConfig, build_reward
from attentive_snake.games.snake.types import Action, GameState, Position, StepEvent, StepResult

_DIGITS = 9  # hides float noise such as -0.060000000000000005
_SIZE = 5
_BODY: tuple[Position, ...] = ((2, 2), (2, 3), (2, 4))  # head at (2, 2), body below it
_FOOD: Position = (4, 2)


def _state(snake: tuple[Position, ...], food: Position | None, score: int, done: bool = False) -> GameState:
    return GameState(_SIZE, _SIZE, snake, Action.RIGHT, food, score, 10, 0, done)


def _moved(head: Position) -> tuple[Position, ...]:
    return (head, *_BODY[:-1])


def _situations(example_score: int) -> list[tuple[str, GameState, StepResult]]:
    no_food = _state(_BODY, None, 0)

    def before(score: int = 0) -> GameState:
        return _state(_BODY, _FOOD, score)

    def move(event: StepEvent, head: Position, score: int = 0) -> StepResult:
        return StepResult(_state(_moved(head), _FOOD, score), event)

    def eat(score: int) -> tuple[GameState, StepResult]:
        grown = ((3, 2), *_BODY)
        return before(score), StepResult(_state(grown, (0, 0), score + 1), StepEvent.ATE_FOOD)

    def ending(event: StepEvent, head: Position) -> StepResult:
        return StepResult(_state(_moved(head), _FOOD, 0, done=True), event)

    return [
        ("Plain move (ignoring distance to food)", no_food, StepResult(_state(_moved((2, 1)), None, 0), StepEvent.MOVED)),
        ("Move closer to the food", before(), move(StepEvent.MOVED, (3, 2))),
        ("Move farther from the food", before(), move(StepEvent.MOVED, (1, 2))),
        ("Eat food at score 0", *eat(0)),
        (f"Eat food at score {example_score}", *eat(example_score)),
        ("Hit a wall", before(), ending(StepEvent.HIT_WALL, (-1, 2))),
        ("Hit itself", before(), ending(StepEvent.HIT_SELF, (2, 3))),
        ("Starve (too long without food)", before(), ending(StepEvent.STARVED, (2, 1))),
        ("Win (fill the board)", before(), ending(StepEvent.WON, (3, 2))),
    ]


def reward_table(config: RewardConfig, example_score: int = 5) -> list[dict]:
    """Rows ``{"situation", "reward", "breakdown": {term: value}}``, one per kind of step."""
    pipeline = build_reward(config)
    rows = []
    for situation, previous, result in _situations(example_score):
        breakdown = pipeline.breakdown(previous, result)
        breakdown = {term: round(value, _DIGITS) for term, value in breakdown.items()}
        rows.append({"situation": situation, "reward": round(sum(breakdown.values()), _DIGITS), "breakdown": breakdown})
    return rows
