"""Snake's ``GameSpec``: how the generic code builds, configures, shows and continues Snake."""

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.env import SnakeEnv
from attentive_snake.games.snake.reward import SnakeRewardConfig
from attentive_snake.games.snake.reward_table import reward_table
from attentive_snake.games.spec import GameSpec


def make_env(rules: SnakeConfig, reward: SnakeRewardConfig, seed: int | None = None) -> SnakeEnv:
    return SnakeEnv(rules, reward, seed=seed)


GAME = GameSpec(
    name="snake",
    title="Snake",
    description="The classic: steer a growing snake to the food without hitting a wall or itself. "
    "The model sees the whole board as a grid of cells.",
    observation_kind="grid",
    view_kind="board",
    rules_type=SnakeConfig,
    reward_type=SnakeRewardConfig,
    make_env=make_env,
    locked_rules={
        "width": "the board size fixes the network's input size",
        "height": "the board size fixes the network's input size",
        "initial_length": "it changes the situations the parent's weights were trained on",
    },
    reward_table=reward_table,
)
