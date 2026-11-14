"""The Lunar Lander reward: Gymnasium's own reward, scaled, with one optional extra for timeouts."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanderReward:
    """``reward = scale * (gymnasium_reward + timeout on the step the time limit ends the episode)`` (``[reward]`` section).

    Gymnasium's reward already shapes the flight (closer to the pad, slower, upright, legs
    down: more points; engine use costs a little; +100 for coming to rest, -100 for a crash).
    The score shown everywhere is the raw Gymnasium return, whatever these values are.
    """

    scale: float = field(default=0.01, metadata={"help": "Multiplies every reward. Gymnasium's returns run from about -400 to +300; 0.01 brings them near the size of Snake's rewards, so the value loss does not swamp the policy in the shared network."})
    timeout: float = field(default=0.0, metadata={"help": "Extra points (in Gymnasium's units, scaled like the rest) when the time limit ends the episode. A negative value discourages hovering until the time runs out; 0 keeps Gymnasium's reward."})

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")


def lander_reward(config: LanderReward, gymnasium_reward: float, timed_out: bool) -> float:
    """The training reward of one step."""
    return config.scale * (gymnasium_reward + (config.timeout if timed_out else 0.0))
