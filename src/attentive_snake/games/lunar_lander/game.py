"""Lunar Lander's ``GameSpec``."""

from attentive_snake.games.lunar_lander.config import LanderRules
from attentive_snake.games.lunar_lander.env import LunarLanderEnv
from attentive_snake.games.lunar_lander.metrics import LANDER_METRICS
from attentive_snake.games.lunar_lander.reward import LanderReward
from attentive_snake.games.spec import GameSpec

OBSERVATION_LABELS = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular velocity",
    "left leg contact",
    "right leg contact",
)


def make_env(rules: LanderRules, reward: LanderReward, seed: int | None = None) -> LunarLanderEnv:
    return LunarLanderEnv(rules, reward, seed)


GAME = GameSpec(
    name="lunar_lander",
    title="Lunar Lander",
    description="Fire the main and side engines to bring a lander down gently between the flags "
    "(Gymnasium's LunarLander-v3). The model sees 8 numbers: position, velocity, angle, spin and leg contacts.",
    observation_kind="vector",
    view_kind="image",
    rules_type=LanderRules,
    reward_type=LanderReward,
    make_env=make_env,
    observation_labels=OBSERVATION_LABELS,
    episode_metrics=LANDER_METRICS,
)
