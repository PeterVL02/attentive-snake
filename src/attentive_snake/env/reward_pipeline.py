"""Composition of reward terms and the fluent builder that assembles them."""

from dataclasses import dataclass

from attentive_snake.env.reward_terms import (
    DeathPenalty,
    FoodDistanceShaping,
    FoodReward,
    RewardTerm,
    StarvationPenalty,
    StepCost,
    WinReward,
)
from attentive_snake.game.types import GameState, StepResult


@dataclass(frozen=True)
class RewardPipeline:
    """Ordered terms whose outputs are summed. Itself a valid RewardTerm."""

    terms: tuple[RewardTerm, ...] = ()
    name: str = "pipeline"

    def __call__(self, previous: GameState, result: StepResult) -> float:
        return float(sum(term(previous, result) for term in self.terms))

    def breakdown(self, previous: GameState, result: StepResult) -> dict[str, float]:
        """Contribution of each term by name (terms sharing a name are summed)."""
        out: dict[str, float] = {}
        for term in self.terms:
            out[term.name] = out.get(term.name, 0.0) + term(previous, result)
        return out


class RewardBuilder:
    """Fluent builder for a RewardPipeline. Terms are applied in the order added."""

    def __init__(self) -> None:
        self._terms: list[RewardTerm] = []

    def add(self, term: RewardTerm) -> "RewardBuilder":
        """Append any custom RewardTerm."""
        self._terms.append(term)
        return self

    def food(self, value: float, growth: float = 0.0) -> "RewardBuilder":
        return self.add(FoodReward(value, growth))

    def death(self, value: float) -> "RewardBuilder":
        return self.add(DeathPenalty(value))

    def win(self, value: float) -> "RewardBuilder":
        return self.add(WinReward(value))

    def starvation(self, value: float) -> "RewardBuilder":
        return self.add(StarvationPenalty(value))

    def step_cost(self, value: float) -> "RewardBuilder":
        return self.add(StepCost(value))

    def food_distance(self, closer: float = 0.0, farther: float = 0.0) -> "RewardBuilder":
        return self.add(FoodDistanceShaping(closer, farther))

    def build(self) -> RewardPipeline:
        return RewardPipeline(tuple(self._terms))
