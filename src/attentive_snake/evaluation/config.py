"""The ``[evaluation]`` section of an experiment: whether and how a run evaluates its policy while it trains.

Its own section, not part of ``[train]``: none of it changes what the network learns (the
PPO settings do), any of it may change when a run is resumed or forked, and
``scripts/evaluate.py`` and the app use it as the defaults for evaluating a checkpoint
after training. Pure data plus validation. Field descriptions live in ``metadata["help"]``.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalConfig:
    interval: int = field(default=0, metadata={"help": "Evaluate the model every this many env steps while training (and once at the end): it plays full games without exploration (always its most likely action), a checkpoint is saved at that step, and the statistics are appended to the run's evaluations.jsonl. 0 turns evaluation off."})
    games: int = field(default=100, metadata={"help": "How many complete games each evaluation plays. More games give steadier numbers but take longer."})
    seed: int = field(default=1000, metadata={"help": "Seed of the evaluation games (independent of the training seed). The same checkpoint with the same seed and number of games always gives the same numbers, so a recorded evaluation can be reproduced later."})
    max_episode_steps: int = field(default=10_000, metadata={"help": "Safety limit for evaluation games: a game still going after this many steps is stopped and counted with the outcome 'step_limit'. Needed for games without their own time limit, where a model that never makes progress would otherwise play forever."})

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError("interval must be 0 (off) or positive.")
        if self.games < 1:
            raise ValueError("games must be positive.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")

    @property
    def enabled(self) -> bool:
        return self.interval > 0
