"""The two-player Atari rules: which title, and how the emulator is stepped and its screen shown to the model."""

from dataclasses import dataclass, field

from attentive_snake.games.atari_versus.titles import DEFAULT_TITLE, TITLE_KEYS, get_title

MIN_SCREEN_SIZE = 36  # the smallest picture the image models' convolutions can read


@dataclass(frozen=True)
class VersusRules:
    """Title, emulator settings and screen preprocessing of one two-player Atari game (``[game]`` section)."""

    title: str = field(default=DEFAULT_TITLE, metadata={"choices": TITLE_KEYS, "help": "Which two-player Atari 2600 game to play. Pong: two paddles, first to 21 points. Combat: Tank: two tanks shooting at each other in a maze, shots bounce off the walls, the game ends on a timer. Combat: Plane: two biplanes with guided missiles, on a timer. Boxing: two boxers seen from above, points per punch, on a timer. While training, the model plays both players (self-play)."})
    frame_skip: int = field(default=4, metadata={"help": "How many screen frames each chosen action is held for. The console draws 60 frames a second; with 4 the players decide 15 times a second and the model sees every fourth frame (the brighter of the last two at each pixel). One step of a match is this many frames."})
    sticky_actions: float = field(default=0.25, metadata={"help": "Chance, on every screen frame, that the console ignores a player's new action and repeats its previous one. It makes the game unpredictable, so the players cannot just memorise one sequence of button presses. 0.25 is the standard of the single-player Atari environments; with 0 a match depends only on the two players' actions."})
    full_action_space: bool = field(default=True, metadata={"help": "Offer all 18 joystick actions (8 directions, each with and without the fire button, plus fire alone and doing nothing) in every title, even where several do the same thing. Off: only the actions the title distinguishes (6 in Pong), which is easier to learn."})
    max_episode_steps: int = field(default=27000, metadata={"help": "Cut a match off as 'timeout' after this many steps (27000 steps of 4 frames = 30 minutes of play). Combat and Boxing end on their own timer after about 2 minutes; Pong only when one side reaches 21, which two good players may take a long time to do."})
    screen_size: int = field(default=84, metadata={"help": "Side length in pixels of the square picture the model sees; the screen is shrunk to it. 84 is the usual size. Smaller is faster but can lose small objects such as Pong's ball or Combat's shots. At least 36."})
    grayscale: bool = field(default=True, metadata={"help": "Show the model brightness only (one number per pixel) instead of red, green and blue (three). Either way the model gets one more plane per frame that says which of the two players it controls (0 for the first, 1 for the second), because both players see the same screen."})

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
