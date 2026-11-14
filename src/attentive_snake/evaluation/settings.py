"""``EvalSettings``: everything that determines an evaluation's numbers, recorded with every evaluation.

Greedy play is deterministic, so the same checkpoint with equal settings gives the same
statistics. ``parallel_games`` is part of it because it fixes which games share a batch of
the network (batch composition can change float results in the last bits, and with them
an argmax now and then); ``device`` for the same reason across CPU and GPU.
"""

from dataclasses import asdict, dataclass
from typing import Any

from attentive_snake.evaluation.config import EvalConfig

PARALLEL_GAMES = 32
POLICY = "greedy"  # the most likely action of every step; no sampling


@dataclass(frozen=True)
class EvalSettings:
    games: int
    seed: int
    max_episode_steps: int
    parallel_games: int = PARALLEL_GAMES
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.games < 1 or self.max_episode_steps < 1 or self.parallel_games < 1:
            raise ValueError("games, max_episode_steps and parallel_games must be positive.")

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "policy": POLICY}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EvalSettings":
        return cls(**{key: data[key] for key in ("games", "seed", "max_episode_steps", "parallel_games", "device")})

    def same_games(self, other: "EvalSettings") -> bool:
        """Whether both play the same games the same way (everything but the device)."""
        return (self.games, self.seed, self.max_episode_steps, self.parallel_games) == (
            other.games, other.seed, other.max_episode_steps, other.parallel_games
        )


def settings_from_config(config: EvalConfig, device: str, **changes: int | None) -> EvalSettings:
    """The settings of ``config`` on ``device``, with the given (not ``None``) changes, e.g. ``games=10``."""
    values = {"games": config.games, "seed": config.seed, "max_episode_steps": config.max_episode_steps}
    values |= {key: value for key, value in changes.items() if value is not None}
    return EvalSettings(**values, device=device)
