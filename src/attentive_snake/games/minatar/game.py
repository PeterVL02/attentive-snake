"""MinAtar's ``GameSpec``: one game for every title (``MinAtarRules.title``)."""

from attentive_snake.games.minatar.config import MinAtarRules
from attentive_snake.games.minatar.env import MinAtarEnv
from attentive_snake.games.minatar.reward import MinAtarReward
from attentive_snake.games.minatar.reward_table import reward_table
from attentive_snake.games.minatar.titles import TITLES
from attentive_snake.games.spec import GameSpec


def make_env(rules: MinAtarRules, reward: MinAtarReward, seed: int | None = None) -> MinAtarEnv:
    return MinAtarEnv(rules, reward, seed)


GAME = GameSpec(
    name="minatar",
    title="MinAtar",
    description=f"Small 10 x 10 versions of Atari games ({', '.join(title.label for title in TITLES)}; Young & Tian 2019): "
    "every cell says which kinds of objects are in it, one channel per kind. Choose the title under the game rules. "
    "MinAtar's trail channels show where things were a step earlier; turn them off (game.trails) and the model needs memory.",
    observation_kind="grid",
    view_kind="board",
    rules_type=MinAtarRules,
    reward_type=MinAtarReward,
    make_env=make_env,
    reward_table=reward_table,
    locked_rules={
        "title": "the parent's weights were trained on that title",
        "trails": "it changes the number of channels the model sees, so the parent's weights would not fit",
        "full_action_space": "it changes the number of actions, so the parent's outputs would not fit",
    },
    # Of the default rules (Breakout's own actions). Other titles and full_action_space name
    # their actions themselves (MinAtarEnv.action_labels, read by watch.action_names).
    action_labels=("nothing", "left", "right"),
)
