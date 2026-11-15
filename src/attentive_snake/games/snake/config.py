"""Configuration of the snake game rules. Nothing about rewards, models or rendering."""

from dataclasses import dataclass, field

FOOD_DISTRIBUTIONS = ("uniform", "center")


@dataclass(frozen=True)
class SnakeConfig:
    """Rules of one snake game.

    Field descriptions live in each field's ``metadata["help"]``. ``None`` for
    ``max_steps_without_food`` disables the starvation limit (classic rules,
    used for human play).
    """

    width: int = field(default=10, metadata={"help": "Number of grid columns. Bigger boards are harder to learn: more cells to look at and longer trips to the food."})
    height: int = field(default=10, metadata={"help": "Number of grid rows. Bigger boards are harder to learn: more cells to look at and longer trips to the food."})
    initial_length: int = field(default=3, metadata={"help": "How many cells long the snake is at the start of every game (at most the board width)."})
    max_steps_without_food: int | None = field(default=None, metadata={"help": "End the game as 'starved' after this many steps without eating, so a policy that wanders forever gets a penalty and the episode ends. Empty means no limit."})
    food_distribution: str = field(default="uniform", metadata={"choices": FOOD_DISTRIBUTIONS, "help": "Where new apples appear: 'uniform' puts them on any free cell with equal chance, 'center' prefers cells near the middle of the board (see food_spread). Centred apples are a gentler start: they are reachable with short trips."})
    food_spread: float = field(default=2.0, metadata={"help": "How tightly apples gather around the middle when food_distribution is 'center': the standard deviation in cells (about two thirds of the apples land within this distance). Bigger is closer to uniform. Ignored for 'uniform'."})
    food_count: int = field(default=1, metadata={"help": "How many apples are on the board at once. When one is eaten a new one appears, so there are always this many while there is room. More apples means shorter trips and easier points."})

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Grid must be at least 2x2.")
        if not 1 <= self.initial_length <= self.width:
            raise ValueError("initial_length must be between 1 and the grid width.")
        if self.max_steps_without_food is not None and self.max_steps_without_food < 1:
            raise ValueError("max_steps_without_food must be positive or None.")
        if self.food_distribution not in FOOD_DISTRIBUTIONS:
            raise ValueError(f"food_distribution must be one of {FOOD_DISTRIBUTIONS}.")
        if not self.food_spread > 0:
            raise ValueError("food_spread must be positive.")
        if self.food_count < 1:
            raise ValueError("food_count must be at least 1.")
