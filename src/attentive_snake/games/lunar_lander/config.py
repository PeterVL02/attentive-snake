"""The Lunar Lander rules: the knobs of Gymnasium's ``LunarLander-v3`` worth changing, and its time limit."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanderRules:
    """Physics and episode length of one Lunar Lander game (``[game]`` section)."""

    gravity: float = field(default=-10.0, metadata={"help": "Downward acceleration of the moon (Gymnasium allows between -12 and 0). Closer to 0 makes the lander easier to slow down."})
    enable_wind: bool = field(default=False, metadata={"help": "Blow random wind and turbulence at the lander, so the same actions no longer give the same flight. Makes the game much harder."})
    wind_power: float = field(default=15.0, metadata={"help": "Strongest sideways wind push when wind is enabled (0 to 20)."})
    turbulence_power: float = field(default=1.5, metadata={"help": "Strongest rotating gust when wind is enabled (0 to 2)."})
    max_episode_steps: int = field(default=1000, metadata={"help": "Cut the episode off as 'timeout' after this many steps (Gymnasium's limit is 1000). A lander that hovers forever ends here; the trainer treats the cut like a normal ending."})
    continuous: bool = field(default=False, metadata={"help": "Control the engines with two throttle numbers between -1 and 1 instead of choosing one of four buttons: the first sets the main engine (off at 0 or below, half to full power from 0 to 1), the second the side engines (left below -0.5, right above 0.5, off in between; more power further out). Only PPO can learn such actions, and neither the world-model kinds (jepa...) nor env.history_actions work with them."})
    hide_velocities: bool = field(default=False, metadata={"help": "Show the model zeros instead of the three velocity numbers (x velocity, y velocity, spin). It then has to work them out from how position and angle change, so only a model with a history of past frames can do well. The physics, score and reward still use the true state."})

    def __post_init__(self) -> None:
        if not -12.0 < self.gravity < 0.0:
            raise ValueError("gravity must be between -12 and 0 (exclusive).")
        if not 0.0 <= self.wind_power <= 20.0:
            raise ValueError("wind_power must be between 0 and 20.")
        if not 0.0 <= self.turbulence_power <= 2.0:
            raise ValueError("turbulence_power must be between 0 and 2.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
