import pytest

from attentive_snake.env.reward import RewardConfig, build_reward
from attentive_snake.env.reward_pipeline import RewardBuilder, RewardPipeline
from attentive_snake.env.reward_terms import (
    DeathPenalty,
    FoodDistanceShaping,
    FoodReward,
    StarvationPenalty,
    StepCost,
    WinReward,
)
from attentive_snake.game.types import Action, GameState, StepEvent, StepResult


def state(snake, food, score=0):
    return GameState(8, 8, tuple(snake), Action.RIGHT, food, score, 0, 0, False)


PREV = state([(2, 2), (1, 2)], (5, 2), score=3)
NEW = state([(3, 2), (2, 2)], (5, 2), score=3)


def res(event, new=NEW):
    return StepResult(new, event)


def test_food_growth():
    t = FoodReward(1.0, growth=0.5)
    assert t(PREV, res(StepEvent.ATE_FOOD)) == pytest.approx(2.5)  # 1 + 0.5 * 3
    assert t(state([(2, 2)], (5, 2), score=0), res(StepEvent.ATE_FOOD)) == 1.0
    assert t(PREV, res(StepEvent.MOVED)) == 0.0


def test_event_terms():
    assert DeathPenalty(-2)(PREV, res(StepEvent.HIT_WALL)) == -2
    assert DeathPenalty(-2)(PREV, res(StepEvent.HIT_SELF)) == -2
    assert DeathPenalty(-2)(PREV, res(StepEvent.WON)) == 0
    assert WinReward(9)(PREV, res(StepEvent.WON)) == 9
    assert WinReward(9)(PREV, res(StepEvent.MOVED)) == 0
    assert StarvationPenalty(-3)(PREV, res(StepEvent.STARVED)) == -3
    assert StarvationPenalty(-3)(PREV, res(StepEvent.HIT_WALL)) == 0


def test_step_cost_only_on_non_ending():
    t = StepCost(-0.1)
    assert t(PREV, res(StepEvent.MOVED)) == -0.1
    assert t(PREV, res(StepEvent.ATE_FOOD)) == -0.1
    for ev in (StepEvent.HIT_WALL, StepEvent.HIT_SELF, StepEvent.WON, StepEvent.STARVED):
        assert t(PREV, res(ev)) == 0.0


def test_distance_shaping():
    t = FoodDistanceShaping(closer=0.3, farther=-0.2)
    assert t(PREV, res(StepEvent.MOVED)) == 0.3
    away = state([(1, 2), (2, 2)], (5, 2))
    assert t(PREV, res(StepEvent.MOVED, away)) == -0.2
    equal = state([(3, 3), (2, 3)], (5, 2))
    assert t(PREV, res(StepEvent.MOVED, equal)) == 0.0
    assert t(PREV, res(StepEvent.ATE_FOOD)) == 0.0
    assert t(PREV, res(StepEvent.MOVED, state([(3, 2)], None))) == 0.0


def test_builder_pipeline_and_breakdown():
    pipe = (RewardBuilder().food(1.0, growth=0.1).death(-1.0).win(10.0).starvation(-1.0)
            .step_cost(-0.01).food_distance(closer=0.1, farther=-0.1).build())
    assert isinstance(pipe, RewardPipeline) and len(pipe.terms) == 6
    r = res(StepEvent.ATE_FOOD)
    assert pipe(PREV, r) == pytest.approx(1.3 - 0.01)
    assert pipe.breakdown(PREV, r) == pytest.approx(
        {"food": 1.3, "death": 0, "win": 0, "starvation": 0, "step_cost": -0.01,
         "food_distance": 0})
    assert pipe(PREV, res(StepEvent.WON)) == 10.0


def test_builder_add_custom_term():
    class Const:
        name = "const"

        def __call__(self, previous, result):
            return 5.0

    pipe = RewardBuilder().add(Const()).step_cost(-1).build()
    assert pipe(PREV, res(StepEvent.MOVED)) == 4.0
    assert pipe.breakdown(PREV, res(StepEvent.MOVED)) == {"const": 5.0, "step_cost": -1}
    assert RewardBuilder().build()(PREV, res(StepEvent.MOVED)) == 0.0


def test_build_reward_skips_zero_terms_and_defaults():
    default = build_reward(RewardConfig())
    assert [t.name for t in default.terms] == ["food", "death", "win", "starvation", "step_cost"]
    assert default(PREV, res(StepEvent.ATE_FOOD)) == pytest.approx(0.99)
    assert default(PREV, res(StepEvent.MOVED)) == pytest.approx(-0.01)
    assert default(PREV, res(StepEvent.WON)) == 1.0
    sparse = build_reward(RewardConfig(food=1.0, death=0, win=0, starved=0, step=0))
    assert [t.name for t in sparse.terms] == ["food"]
    grow = build_reward(RewardConfig(food=0.0, food_growth=0.2, step=0))
    assert "food" in [t.name for t in grow.terms]
    assert grow(PREV, res(StepEvent.ATE_FOOD)) == pytest.approx(0.6)
