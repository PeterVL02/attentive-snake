"""Atari's ``GameSpec``: one game for every title (``AtariRules.title``)."""

from attentive_snake.ale.actions import FULL_ACTION_LABELS
from attentive_snake.games.atari.config import AtariRules
from attentive_snake.games.atari.env import AtariEnv
from attentive_snake.games.atari.metrics import ATARI_METRICS
from attentive_snake.games.atari.reward import AtariReward
from attentive_snake.games.atari.titles import TITLES
from attentive_snake.games.spec import GameSpec


def make_env(rules: AtariRules, reward: AtariReward, seed: int | None = None) -> AtariEnv:
    return AtariEnv(rules, reward, seed)


GAME = GameSpec(
    name="atari",
    title="Atari",
    description=f"Atari 2600 games ({', '.join(title.label for title in TITLES)}) played from the screen with a joystick: "
    "the model sees a small picture of the screen and picks one of 18 joystick actions. Choose the title under the game rules. "
    "One picture does not show how things move, so the model is given the last few frames (history).",
    observation_kind="image",
    view_kind="image",
    rules_type=AtariRules,
    reward_type=AtariReward,
    make_env=make_env,
    episode_metrics=ATARI_METRICS,
    locked_rules={
        "title": "the parent's weights were trained on that title",
        "screen_size": "it changes the size of the picture, so the parent's weights would not fit",
        "grayscale": "it changes the number of colour channels, so the parent's weights would not fit",
        "full_action_space": "it changes the number of actions, so the parent's outputs would not fit",
        "continuous": "it changes what the network outputs (action probabilities or three joystick numbers), so the parent's outputs would not fit",
    },
    continuous_actions=lambda rules: rules.continuous,  # the env then names the stick's numbers itself (AtariEnv)
    # Of the default rules (full_action_space). An env with a title's own shorter list names its
    # actions itself (AtariEnv.action_labels, read by watch.action_names).
    action_labels=FULL_ACTION_LABELS,
)
