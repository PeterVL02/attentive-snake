"""DQN's own settings (the ``[dqn]`` section). Pure data, no behaviour beyond validation."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DqnConfig:
    """Settings of DQN (Double DQN by default). Field descriptions live in ``metadata["help"]``."""

    buffer_size: int = field(default=100_000, metadata={"help": "How many past transitions the replay buffer keeps (oldest are overwritten). Memory: about 2 x buffer_size x (numbers in one observation) x 4 bytes, e.g. 100,000 transitions of an 8x8 Snake board (192 numbers) take ~150 MB, of Lunar Lander (8 numbers) ~6 MB; history frames multiply it. The buffer is not saved in checkpoints: a resumed or forked run starts with an empty one."})
    batch_size: int = field(default=64, metadata={"help": "Transitions sampled from the replay buffer for each gradient step."})
    learning_starts: int = field(default=10_000, metadata={"help": "Gradient steps begin once the replay buffer holds this many transitions (at least batch_size). Counts transitions in the buffer, so a resumed or forked run first refills its empty buffer this far."})
    train_frequency: int = field(default=4, metadata={"help": "Env steps (summed over all parallel games) between rounds of gradient steps; independent of num_envs. With 16 games and 4, every step of the 16 games is followed by 4 rounds. Lower means more learning per experience (and slower training)."})
    gradient_steps: int = field(default=1, metadata={"help": "Gradient steps per round (see train_frequency)."})
    target_update_interval: int = field(default=10_000, metadata={"help": "Env steps between copies of the online network into the target network, which supplies the bootstrap values. Larger is more stable but slower to propagate what was learned."})
    epsilon_start: float = field(default=1.0, metadata={"help": "Exploration at the start: the probability of a random action instead of the one with the highest Q-value. A forked run starts again from this value (its step count restarts at 0); lower it in the fork dialog to explore less."})
    epsilon_end: float = field(default=0.05, metadata={"help": "Exploration after the decay (see epsilon_fraction), kept until the end of training. Evaluation and watching never explore."})
    epsilon_fraction: float = field(default=0.1, metadata={"help": "Share of total_env_steps over which epsilon falls linearly from epsilon_start to epsilon_end."})
    max_grad_norm: float = field(default=10.0, metadata={"help": "Gradients are scaled down if their overall size exceeds this value, which protects against occasional huge updates."})
    double: bool = field(default=True, metadata={"help": "Double DQN: the online network picks the next action and the target network values it, which curbs the overestimation of plain DQN (off: the target network's highest Q-value)."})

    def __post_init__(self) -> None:
        for name in ("buffer_size", "batch_size", "train_frequency", "gradient_steps", "target_update_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative.")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size must not exceed buffer_size.")
        for name in ("epsilon_start", "epsilon_end"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if not 0.0 <= self.epsilon_fraction <= 1.0:
            raise ValueError("epsilon_fraction must be in [0, 1].")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")
