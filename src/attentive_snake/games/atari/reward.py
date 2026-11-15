"""The Atari reward: the game's own points, clipped to their sign and scaled, plus an optional cost per life lost."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AtariReward:
    """``reward = scale * (sign(points) if clip else points) + life_lost per life lost`` (``[reward]`` section).

    The score shown everywhere is the raw sum of the game's points, whatever these values are.
    """

    clip: bool = field(default=True, metadata={"help": "Reward +1 for any step that gains points and -1 for any that loses points, however many. Atari titles score on very different scales (1 per point in Pong, hundreds per item elsewhere); clipping makes one set of training settings work for all of them, at the price that the model cannot tell a big gain from a small one."})
    scale: float = field(default=1.0, metadata={"help": "Multiplies the (clipped or raw) points. With clip off, set it so that typical rewards are around 1 in size (0.01 for a game scoring in hundreds)."})
    life_lost: float = field(default=0.0, metadata={"help": "Added once for every life lost (not scaled, not clipped). Many titles take a life without taking points, so the model only learns that dying is bad through the points it misses later; a negative value such as -1 says so directly. 0 keeps the game's own reward. Pong has no lives."})

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")


def atari_reward(config: AtariReward, points: float, lives_lost: int) -> float:
    """The training reward of one step that scored ``points`` and cost ``lives_lost`` lives."""
    kept = float((points > 0) - (points < 0)) if config.clip else points
    return config.scale * kept + config.life_lost * lives_lost
