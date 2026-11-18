"""Making the Arcade Learning Environment emulator of a title (Gymnasium's ``ALE/<Title>-v5``).

Gymnasium and ``ale-py`` are imported here, on first use, so that importing the game package
costs nothing. The Gymnasium env is made frame by frame (``frameskip=1``) and without a time
limit: ``env.AtariEnv`` holds actions over frames and counts steps itself.
"""

from attentive_snake.games.atari.config import AtariRules
from attentive_snake.games.atari.titles import get_title


def make_emulator(rules: AtariRules):
    """The unwrapped Gymnasium Atari env of ``rules.title``: ``.ale`` is the emulator, ``.reset(seed=)`` reseeds it."""
    import ale_py
    import gymnasium

    gymnasium.register_envs(ale_py)
    env = gymnasium.make(
        get_title(rules.title).env_id,
        frameskip=1,
        repeat_action_probability=rules.sticky_actions,
        full_action_space=rules.full_action_space,
        max_num_frames_per_episode=0,  # no limit of the emulator's: AtariRules.max_episode_steps is ours
        disable_env_checker=True,
        **_chosen_variant(rules),
    )
    return env.unwrapped


def _chosen_variant(rules: AtariRules) -> dict[str, int]:
    """The mode and difficulty the rules choose; one left unset is not passed, so the title's default applies."""
    chosen = {"mode": rules.mode, "difficulty": rules.difficulty}
    return {name: value for name, value in chosen.items() if value is not None}
