"""The Atari reward: the game's own points, optionally clipped to their sign, scaled, plus an optional cost per life lost."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AtariReward:
    """``reward = scale * (sign(points) if clip else points) + life_lost per life lost`` (``[reward]`` section).

    The score shown everywhere is the raw sum of the game's points, whatever these values are.
    """

    clip: bool = field(default=False, metadata={"help": "Off (the default): the model learns from the game's real points, so it can tell a big gain from a small one. On: reward +1 for any step that gains points and -1 for any that loses points, however many. Atari titles score on very different scales (1 per point in Pong, hundreds per item elsewhere); clipping makes one set of training settings work for all of them, at the price of that distinction."})
    scale: float = field(default=1.0, metadata={"help": "Multiplies the points (raw, or clipped when clip is on). The default 1 keeps the game's own numbers, which can be hundreds per step in some titles; PPO's Atari defaults rescale them anyway (ppo.normalize_returns); DQN does not, so for DQN a smaller value such as 0.01 brings those rewards nearer to 1 in size."})
    life_lost: float = field(default=0.0, metadata={"help": "Added once for every life lost (not scaled, not clipped). Many titles take a life without taking points, so the model only learns that dying is bad through the points it misses later; a negative value such as -1 says so directly. 0 keeps the game's own reward. Pong has no lives."})

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")


def atari_reward(config: AtariReward, points: float, lives_lost: int) -> float:
    """The training reward of one step that scored ``points`` and cost ``lives_lost`` lives."""
    kept = float((points > 0) - (points < 0)) if config.clip else points
    return config.scale * kept + config.life_lost * lives_lost
