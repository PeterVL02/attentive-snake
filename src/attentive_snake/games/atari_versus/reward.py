"""The two-player Atari reward: a player's own points, optionally clipped to their sign, and scaled.

The titles here are zero-sum: whatever one player wins the other loses, so a player's points
on a step are its margin on that step (Pong: +1 when it scores, -1 when the other side does).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VersusReward:
    """``reward = scale * (sign(points) if clip else points)`` for each player (``[reward]`` section).

    The score shown everywhere is the raw sum of the player's points (its margin over the other
    player), whatever these values are.
    """

    clip: bool = field(default=False, metadata={"help": "Off (the default): the player learns from its real points, so Boxing's 2 points for a close punch count double. On: reward +1 for any step on which the player gains points and -1 for any on which it loses points, however many."})
    scale: float = field(default=1.0, metadata={"help": "Multiplies the points (raw, or clipped when clip is on)."})

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")


def versus_reward(config: VersusReward, points: float) -> float:
    """The training reward of one player's step that won (or, negative, lost) ``points``."""
    kept = float((points > 0) - (points < 0)) if config.clip else float(points)
    return config.scale * kept
