"""Two-player Atari's ``GameSpec``: one game for every title (``VersusRules.title``), trained by self-play."""

from attentive_snake.ale.actions import FULL_ACTION_LABELS
from attentive_snake.games.atari_versus.config import VersusRules
from attentive_snake.games.atari_versus.env import VersusEnv
from attentive_snake.games.atari_versus.match import VersusMatch
from attentive_snake.games.atari_versus.metrics import VERSUS_METRICS
from attentive_snake.games.atari_versus.reward import VersusReward
from attentive_snake.games.atari_versus.titles import TITLES
from attentive_snake.games.spec import GameSpec


def make_env(rules: VersusRules, reward: VersusReward, seed: int | None = None) -> VersusEnv:
    return VersusEnv(rules, reward, seed)


def make_match(rules: VersusRules, reward: VersusReward, seed: int | None = None) -> VersusMatch:
    return VersusMatch(rules, reward, seed)


GAME = GameSpec(
    name="atari_versus",
    title="Atari two-player",
    description=f"Two-player Atari 2600 games ({', '.join(title.label for title in TITLES)}) learned by self-play: one model "
    "plays both players from the screen, told which one it is by an extra plane of the picture. Choose the title under the "
    "game rules. Evaluation and Watch put the model against a player pressing random buttons, so their win rate and "
    "score (the model's points minus the opponent's) show progress that self-play's own results cannot.",
    observation_kind="image",
    view_kind="image",
    rules_type=VersusRules,
    reward_type=VersusReward,
    make_env=make_env,
    episode_metrics=VERSUS_METRICS,
    locked_rules={
        "title": "the parent's weights were trained on that title",
        "screen_size": "it changes the size of the picture, so the parent's weights would not fit",
        "grayscale": "it changes the number of colour channels, so the parent's weights would not fit",
        "full_action_space": "it changes the number of actions, so the parent's outputs would not fit",
    },
    # Of the default rules (full_action_space). An env with a title's own shorter list names its
    # actions itself (VersusEnv.action_labels, read by watch.action_names).
    action_labels=FULL_ACTION_LABELS,
    search_objective="win_rate",
    seats=2,
    make_match=make_match,
)
