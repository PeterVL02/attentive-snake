"""Hyperparameters of a PPO training run. Pure data, no behaviour beyond validation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Settings for ``Trainer``.

    The number of parallel environments is a property of the ``VecEnv`` passed
    to the trainer, so one update consumes ``rollout_length * env.num_envs``
    environment steps. Field descriptions live in ``metadata["help"]``.
    """

    total_env_steps: int = field(default=5_000_000, metadata={"help": "Training budget: the total number of game steps taken, summed over all parallel environments. Training stops when it is reached; raise it when resuming to train longer."})
    rollout_length: int = field(default=128, metadata={"help": "How many steps each environment plays before the network is updated. Longer rollouts give the learner more context per update but make updates less frequent."})
    num_epochs: int = field(default=4, metadata={"help": "How many times PPO re-uses each collected batch of experience. More epochs squeeze more out of the data but risk overfitting the batch."})
    num_minibatches: int = field(default=4, metadata={"help": "Each epoch splits the batch into this many pieces and takes one gradient step per piece. More pieces means smaller, noisier steps."})
    gamma: float = field(default=0.99, metadata={"help": "Discount factor: how much future reward counts compared with immediate reward. Close to 1 means far-sighted; lower values make the agent short-sighted."})
    gae_lambda: float = field(default=0.95, metadata={"help": "Smoothing in the advantage estimate (GAE). Near 1 uses long real returns (less bias, more noise); near 0 relies on the value network's guesses (more bias, less noise)."})
    bootstrap_truncated: bool = field(default=False, metadata={"help": "When a game is cut short by a limit (a time limit, a starvation limit) rather than ended by its rules, credit the last step with the value the network expects from where the game was cut, as if play had gone on. Off (how runs made before this setting existed train) treats every cut like a real ending, so the values of states near the limit are underestimated."})
    clip_coef: float = field(default=0.2, metadata={"help": "PPO's trust-region size: how far the new policy may move from the old one in a single update. Smaller is safer and slower; larger is faster and can destabilise training."})
    value_coef: float = field(default=0.5, metadata={"help": "Weight of the value-prediction loss relative to the policy loss. Raise it to make the network focus on predicting returns accurately."})
    entropy_coef: float = field(default=0.01, metadata={"help": "Weight of a bonus for keeping the policy random. Higher means more exploration and slower commitment to one behaviour; 0 lets the policy become deterministic quickly."})
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Step size of the Adam optimiser (its initial value when annealing). Too high makes learning unstable; too low makes it crawl."})
    anneal_lr: bool = field(default=True, metadata={"help": "Linearly shrink the learning rate to zero over the run, which settles training towards the end."})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Gradients are scaled down if their overall size exceeds this value, which protects against occasional huge updates."})
    normalize_advantages: bool = field(default=True, metadata={"help": "Rescale advantages to zero mean and unit variance within each minibatch, so the update size does not depend on the reward scale."})
    target_kl: float | None = field(default=None, metadata={"help": "Optional early stop: if the policy changes more than this (approximate KL divergence) during an update, the remaining epochs are skipped. Empty means never stop early."})
    device: str = field(default="auto", metadata={"help": "Where the network runs: 'auto' picks the GPU if there is one, otherwise 'cpu'. Any torch device string such as 'cuda:0' also works."})
    seed: int = field(default=0, metadata={"help": "Random seed for the network initialisation, the games and the minibatch shuffling. Change it to get an independent repeat of the experiment."})
    log_interval: int = field(default=10, metadata={"help": "Record the training metrics every this many updates."})
    checkpoint_dir: Path | None = field(default=None, metadata={"help": "Folder where checkpoints are saved. Set automatically to the run folder; empty disables checkpoints."})
    checkpoint_interval: int = field(default=100, metadata={"help": "Save a checkpoint every this many updates (and at the end of training)."})
    keep_checkpoints: bool = field(default=True, metadata={"help": "Keep every saved checkpoint. When off, older checkpoints of the run are deleted each time a new one is saved, so only the latest remains (saves disk space, but older checkpoints can then no longer be watched or continued from)."})

    def __post_init__(self) -> None:
        positive_ints = (
            "total_env_steps",
            "rollout_length",
            "num_epochs",
            "num_minibatches",
            "log_interval",
            "checkpoint_interval",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        for name in ("gamma", "gae_lambda"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in ("clip_coef", "learning_rate", "max_grad_norm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("Loss coefficients must be non-negative.")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive or None.")
