"""Configuration of the snake game rules. Nothing about rewards, models or rendering."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameConfig:
    """Rules of one snake game.

    Field descriptions live in each field's ``metadata["help"]``. ``None`` for
    ``max_steps_without_food`` disables the starvation limit (classic rules,
    used for human play).
    """

    width: int = field(default=10, metadata={"help": "Number of grid columns. Bigger boards are harder to learn: more cells to look at and longer trips to the food."})
    height: int = field(default=10, metadata={"help": "Number of grid rows. Bigger boards are harder to learn: more cells to look at and longer trips to the food."})
    initial_length: int = field(default=3, metadata={"help": "How many cells long the snake is at the start of every game (at most the board width)."})
    max_steps_without_food: int | None = field(default=None, metadata={"help": "End the game as 'starved' after this many steps without eating, so a policy that wanders forever gets a penalty and the episode ends. Empty means no limit."})

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Grid must be at least 2x2.")
        if not 1 <= self.initial_length <= self.width:
            raise ValueError("initial_length must be between 1 and the grid width.")
        if self.max_steps_without_food is not None and self.max_steps_without_food < 1:
            raise ValueError("max_steps_without_food must be positive or None.")
