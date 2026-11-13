"""Reward configuration (flat, TOML-friendly) and its translation into a pipeline."""

from dataclasses import dataclass, field

from attentive_snake.env.reward_pipeline import RewardBuilder, RewardPipeline
from attentive_snake.env.reward_terms import RewardTerm
from attentive_snake.game.types import GameState, StepResult

__all__ = ["RewardConfig", "build_reward", "resolve_reward", "RewardBuilder", "RewardPipeline",
           "RewardTerm"]


@dataclass(frozen=True)
class RewardConfig:
    """Reward coefficients; a term whose coefficients are all zero is not added.

    The food reward is ``food + food_growth * score_before_eating``; ``step`` is
    also added on a food step. Field descriptions live in ``metadata["help"]``.
    """

    food: float = field(default=1.0, metadata={"help": "Reward for eating a piece of food. The main learning signal: raise it to make eating matter more compared with the penalties."})
    food_growth: float = field(default=0.0, metadata={"help": "Extra food reward for every point of score already earned, so later food pays more (food + food_growth * score). Above 0 the reward grows with how well the snake is doing."})
    death: float = field(default=-1.0, metadata={"help": "Reward for hitting a wall or the snake's own body. Make it more negative to make the snake more careful."})
    win: float = field(default=1.0, metadata={"help": "Reward for filling the whole board. Very rare on large boards, so it mostly matters on small ones."})
    starved: float = field(default=-1.0, metadata={"help": "Reward when the game is cut off for taking too many steps without food (see max_steps_without_food in the game section). Usually negative to discourage wandering."})
    step: float = field(default=-0.01, metadata={"help": "Small cost added on every step that does not end the game. A negative value pushes the snake to reach food quickly; 0 removes the pressure."})
    closer_to_food: float = field(default=0.0, metadata={"help": "Shaping bonus for a plain move that brings the head nearer to the food (Manhattan distance). A positive value gives hints, but can teach the snake to chase food rather than survive."})
    farther_from_food: float = field(default=0.0, metadata={"help": "Shaping value for a plain move that takes the head farther from the food. Use a negative value to penalise moving away."})


def build_reward(config: RewardConfig) -> RewardPipeline:
    """Create the pipeline described by ``config``."""
    builder = RewardBuilder()
    if config.food or config.food_growth:
        builder.food(config.food, config.food_growth)
    if config.death:
        builder.death(config.death)
    if config.win:
        builder.win(config.win)
    if config.starved:
        builder.starvation(config.starved)
    if config.step:
        builder.step_cost(config.step)
    if config.closer_to_food or config.farther_from_food:
        builder.food_distance(config.closer_to_food, config.farther_from_food)
    return builder.build()


def resolve_reward(reward: "RewardConfig | RewardTerm | None") -> RewardTerm:
    """Accept a config (built), a ready pipeline/term/callable (used as is) or None (defaults)."""
    if reward is None:
        return build_reward(RewardConfig())
    if isinstance(reward, RewardConfig):
        return build_reward(reward)
    return reward
