"""The Atari rules: which title, and how the emulator is stepped and its screen shown to the model."""

from dataclasses import dataclass, field

from attentive_snake.games.atari.titles import DEFAULT_TITLE, TITLE_KEYS, get_title

MIN_SCREEN_SIZE = 36  # the smallest picture the image models' convolutions can read


@dataclass(frozen=True)
class AtariRules:
    """Title, emulator settings and screen preprocessing of one Atari game (``[game]`` section)."""

    title: str = field(default=DEFAULT_TITLE, metadata={"choices": TITLE_KEYS, "help": "Which Atari 2600 game to play. In Pong the ball's direction, and in most of the others what moves and how fast, cannot be read from one screen picture: the model needs the last few frames (env.history_length) to play well."})
    frame_skip: int = field(default=4, metadata={"help": "How many screen frames each chosen action is held for. The console draws 60 frames a second; with 4 the model decides 15 times a second and sees every fourth frame (the brighter of the last two at each pixel, because some games draw objects only on alternate frames). One training step is this many frames."})
    sticky_actions: float = field(default=0.25, metadata={"help": "Chance, on every screen frame, that the console ignores the new action and repeats the previous one. It makes the game unpredictable, so a model cannot just memorise one sequence of button presses. 0.25 is the standard of the v5 environments. With 0 a title without randomness of its own (Pong) plays out the same way every time, whatever the seed, so the games of an evaluation would all be copies of one game."})
    full_action_space: bool = field(default=True, metadata={"help": "Offer all 18 joystick actions (8 directions, each with and without the fire button, plus fire alone and doing nothing) in every title, even where several do the same thing. The network then has the same outputs for every title, so models are comparable across them. Off: only the actions the title distinguishes (6 in Pong), which is easier to learn."})
    continuous: bool = field(default=False, metadata={"help": "Move the joystick with three numbers instead of choosing one of the joystick actions: how far the stick is pushed (0 to 1), in which direction (an angle from -3.14 to 3.14, 0 = right, 1.57 = up) and the fire button (pressed above 0.5). A push above half way in a direction counts as that direction, so the game still sees one of the 18 joystick actions (full_action_space then makes no difference). This is the Arcade Learning Environment's continuous mode. Only PPO can learn such actions, and neither the world-model kinds (jepa...) nor env.history_actions work with them."})
    max_episode_steps: int = field(default=27000, metadata={"help": "Cut a game off as 'timeout' after this many steps (27000 steps of 4 frames = 30 minutes of play, the usual Atari limit). The trainer treats the cut like a normal ending unless train.bootstrap_truncated is on."})
    screen_size: int = field(default=84, metadata={"help": "Side length in pixels of the square picture the model sees; the 210 x 160 screen is shrunk to it. 84 is the usual size. Smaller is faster but can lose small objects such as Pong's ball. At least 36."})
    grayscale: bool = field(default=True, metadata={"help": "Show the model brightness only (one number per pixel) instead of red, green and blue (three). Colour rarely matters in these games, and grayscale takes a third of the memory."})

    def __post_init__(self) -> None:
        get_title(self.title)
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1.")
        if not 0.0 <= self.sticky_actions < 1.0:
            raise ValueError("sticky_actions must be at least 0 and below 1.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
        if self.screen_size < MIN_SCREEN_SIZE:
            raise ValueError(f"screen_size must be at least {MIN_SCREEN_SIZE} (the convolutions of the image models need that much).")
