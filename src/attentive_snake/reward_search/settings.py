"""``SearchSettings``: the numbers a reward search is run with. Pure data plus validation; help in ``metadata["help"]``."""

import dataclasses
from dataclasses import dataclass, field
from typing import Any

MAX_POPULATION = 32
MAX_GENERATIONS = 100


@dataclass(frozen=True)
class SearchSettings:
    population: int = field(default=4, metadata={"help": "How many runs train side by side in each generation (at least 2). One of them keeps the current reward unchanged; the others each try a changed copy. More members try more rewards per generation but cost proportionally more training."})
    generations: int = field(default=4, metadata={"help": "How many rounds the search runs. After each round the best member's model and reward are the starting point of the next round, so the reward can drift step by step."})
    steps_per_member: int = field(default=50_000, metadata={"help": "Env steps every member trains in one generation, starting from the best model so far. Too few and the members cannot show a difference; the whole search costs population x generations x this many steps."})
    perturbation: float = field(default=0.3, metadata={"help": "How far one generation may move a selected coefficient, as a fraction of its starting size: 0.3 moves death = -1 by up to 0.3 either way. The move is drawn at random for every changed member."})
    max_change: float = field(default=1.0, metadata={"help": "The farthest a coefficient may ever get from its starting value, as a multiple of its starting size: 1.0 keeps death = -1 between -2 and 0 for the whole search. Stops the search from drifting to extreme rewards."})
    allow_sign_change: bool = field(default=False, metadata={"help": "Off: a coefficient keeps its starting sign (a penalty stays a penalty, a bonus stays a bonus; it may reach 0). On: it may cross 0 and turn into the opposite."})
    zero_scale: float = field(default=0.01, metadata={"help": "The size used for a coefficient that starts at 0, which has no size of its own: it moves by up to perturbation x this per generation, never farther than max_change x this from 0, and in either direction (0 has no sign to keep). Set it to the size you would consider a small value for that coefficient."})
    objective: str | None = field(default=None, metadata={"help": "The evaluation statistic that decides which member wins. Only statistics of what happened in the game are offered (never the return, which depends on the reward being searched). Empty: the game's own choice (for Snake the mean score)."})
    evaluation_games: int = field(default=100, metadata={"help": "Greedy games each member plays at the end of its training to be judged. Every member plays the same games (same seed), so differences come from the models, not from luck of the draw; more games give steadier numbers."})
    seed: int = field(default=0, metadata={"help": "Seed of the random changes to the reward. The same seed, settings and starting table always propose the same tables."})

    def __post_init__(self) -> None:
        if not 2 <= self.population <= MAX_POPULATION:
            raise ValueError(f"population must be between 2 and {MAX_POPULATION}.")
        if not 1 <= self.generations <= MAX_GENERATIONS:
            raise ValueError(f"generations must be between 1 and {MAX_GENERATIONS}.")
        if self.steps_per_member < 1:
            raise ValueError("steps_per_member must be positive.")
        for name in ("perturbation", "max_change", "zero_scale"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive.")
        if self.evaluation_games < 1:
            raise ValueError("evaluation_games must be positive.")


def settings_to_json(settings: SearchSettings) -> dict[str, Any]:
    return dataclasses.asdict(settings)


def settings_from_json(data: dict[str, Any]) -> SearchSettings:
    """``TypeError`` for unknown keys, ``ValueError`` for invalid values."""
    return SearchSettings(**data)
