"""The MinAtar reward: the game's own points, scaled, plus an optional amount when the game is lost."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MinAtarReward:
    """``reward = scale * points + game_over on the step the title ends the game`` (``[reward]`` section).

    The score shown everywhere is the raw sum of the game's points, whatever these values are.
    """

    scale: float = field(default=1.0, metadata={"help": "Multiplies the game's points. MinAtar pays small whole numbers (1 per brick, alien, enemy, road crossed or treasure; Seaquest up to 10 more for surfacing with six divers), so 1 keeps them as they are."})
    game_over: float = field(default=0.0, metadata={"help": "Added (not scaled) on the step the game ends by its own rules: the ball lost in Breakout, the player hit in Space Invaders, Seaquest and Asterix. In Freeway the game only ends when its time runs out, so a value here is paid at the end of every Freeway game whatever happened. A negative value tells the model directly that losing is bad; 0 keeps MinAtar's reward. Not paid when max_episode_steps cuts a game off."})

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")


def minatar_reward(config: MinAtarReward, points: float, game_over: bool) -> float:
    """The training reward of one step that scored ``points`` and maybe ended the game."""
    return config.scale * points + (config.game_over if game_over else 0.0)
