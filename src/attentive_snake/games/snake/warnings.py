"""Snake's warning about a step cost that makes dying no worse than staying alive."""

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.reward import SnakeRewardConfig

ROUNDING = 1e-9  # 0.01 / (1 - 0.99) is 0.9999999999999991: "equal" must count


def step_cost_warnings(_rules: SnakeConfig, reward: SnakeRewardConfig, gamma: float) -> tuple[str, ...]:
    """Warn if the discounted living cost without food, ``|step| / (1 - gamma)``, reaches the death penalty."""
    if reward.step >= 0 or reward.death == 0:
        return ()
    if gamma < 1 and abs(reward.step) / (1 - gamma) < abs(reward.death) * (1 - ROUNDING):
        return ()
    cost = "more than" if gamma >= 1 else f"{reward.step / (1 - gamma):.2f} against"
    return (
        f"With a step cost of {reward.step:g} and gamma {gamma:g}, staying alive without finding food costs as much "
        f"as dying ({cost} a death penalty of {reward.death:.2f}). Until the snake has learned to find apples, "
        "nothing rewards staying alive, so early in training it may run straight into a wall; this lasts longer on "
        f"large boards. Lower the step cost (for example to {reward.step / 10:g}), or expect this phase.",
    )
