"""The MinAtar rules: which title, which channels the model sees, and how the game is stepped."""

from dataclasses import dataclass, field

from attentive_snake.games.minatar.titles import DEFAULT_TITLE, TITLE_KEYS, get_title


@dataclass(frozen=True)
class MinAtarRules:
    """Title and settings of one MinAtar game (``[game]`` section)."""

    title: str = field(default=DEFAULT_TITLE, metadata={"choices": TITLE_KEYS, "help": "Which MinAtar game to play: small 10 x 10 versions of Atari games (Breakout, Space Invaders, Freeway, Seaquest, Asterix) where every cell says which kinds of objects are in it."})
    trails: bool = field(default=True, metadata={"help": "Show the model MinAtar's motion channels, as published: 'trail' marks where the ball or a moving object was one step earlier (Breakout, Seaquest, Asterix), Freeway's five trail channels also tell each car's speed, and Space Invaders has two channels saying which way the aliens march. With them one picture shows how everything moves. Off: these channels are removed from the observation (not blanked: the model has fewer inputs), so one picture no longer shows direction or speed and the game needs memory. Then give the model the last few frames (env.history_length above 1) or use a kind that reads them in order (frame_attention, temporal_transformer, lstm)."})
    sticky_action_prob: float = field(default=0.1, metadata={"help": "Chance, on every step, that the game ignores the chosen action and repeats the previous one (MinAtar's own setting, 0.1 by default). It makes the game unpredictable, so a model cannot just memorise one sequence of moves. 0: every game with the same seed plays out the same way."})
    difficulty_ramping: bool = field(default=True, metadata={"help": "Let the game get harder as it goes on, as MinAtar does by default: in Seaquest and Asterix enemies come faster and more often over time, in Space Invaders each new wave of aliens moves faster. Breakout and Freeway do not ramp. Off: the difficulty stays at its starting level."})
    full_action_space: bool = field(default=False, metadata={"help": "Offer all 6 MinAtar actions (nothing, left, up, right, down, fire) in every title, as the MinAtar paper did, even where several do the same thing. The network then has the same outputs for every title. Off: only the actions the title distinguishes (3 in Breakout: nothing, left, right), which is easier to learn."})
    max_episode_steps: int = field(default=10000, metadata={"help": "Cut a game off as 'timeout' after this many steps. MinAtar has no limit of its own except Freeway's (2500 steps, which ends the game normally); a good Breakout player can otherwise play for ever. The trainer treats the cut like a normal ending unless train.bootstrap_truncated is on."})

    def __post_init__(self) -> None:
        get_title(self.title)
        if not 0.0 <= self.sticky_action_prob < 1.0:
            raise ValueError("sticky_action_prob must be at least 0 and below 1.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
