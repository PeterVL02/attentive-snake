"""Settings every training algorithm shares (the ``[train]`` section). Pure data, no behaviour beyond validation.

Each algorithm's own settings live in its own section and dataclass (``[ppo]``: ``PpoConfig``,
``[dqn]``: ``DqnConfig``), chosen by ``algorithm`` through ``attentive_snake.training.algorithms``.
The number of parallel environments is ``[env] num_envs``.
"""

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ALGORITHM = "ppo"


@dataclass(frozen=True)
class TrainConfig:
    """Settings of the training loop shared by every algorithm. Field descriptions live in ``metadata["help"]``."""

    algorithm: str = field(default=DEFAULT_ALGORITHM, metadata={"help": "How the network learns: 'ppo' (on-policy actor-critic: a policy and a value head, learns from fresh experience only) or 'dqn' (off-policy, value-based: one Q-value per action, learns from a replay buffer of past experience). Its own settings are in the section of the same name."})
    total_env_steps: int = field(default=5_000_000, metadata={"help": "Training budget: the total number of game steps taken, summed over all parallel environments. Training stops when it is reached; raise it when resuming to train longer."})
    gamma: float = field(default=0.99, metadata={"help": "Discount factor: how much future reward counts compared with immediate reward. Close to 1 means far-sighted; lower values make the agent short-sighted."})
    bootstrap_truncated: bool = field(default=False, metadata={"help": "When a game is cut short by a limit (a time limit, a starvation limit) rather than ended by its rules, credit the last step with the value the network expects from where the game was cut, as if play had gone on. Off treats every cut like a real ending, so the values of states near the limit are underestimated."})
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Step size of the Adam optimiser (its initial value when annealing). Too high makes learning unstable; too low makes it crawl."})
    anneal_lr: bool = field(default=True, metadata={"help": "Linearly shrink the learning rate to zero over the run, which settles training towards the end."})
    device: str = field(default="auto", metadata={"help": "Where the network runs: 'auto' picks the GPU if there is one, otherwise 'cpu'. Any torch device string such as 'cuda:0' also works."})
    seed: int = field(default=0, metadata={"help": "Random seed for the network initialisation, the games and the algorithm's own randomness (minibatch shuffling, exploration, replay sampling). Change it to get an independent repeat of the experiment."})
    log_interval: int = field(default=50_000, metadata={"help": "Record the training metrics each time the env-step count passes a multiple of this (and at the end)."})
    checkpoint_dir: Path | None = field(default=None, metadata={"help": "Folder where checkpoints are saved. Set automatically to the run folder; empty disables checkpoints."})
    checkpoint_interval: int = field(default=500_000, metadata={"help": "Save a checkpoint each time the env-step count passes a multiple of this (and at the end of training)."})
    keep_checkpoints: bool = field(default=True, metadata={"help": "Keep every saved checkpoint. When off, older checkpoints of the run are deleted each time a new one is saved, so only the latest remains (saves disk space, but older checkpoints can then no longer be watched or continued from)."})

    def __post_init__(self) -> None:
        # Imported here: the registry imports the algorithms, which import this module.
        from attentive_snake.training.algorithms import algorithm_names

        if self.algorithm not in algorithm_names():
            raise ValueError(f"unknown algorithm {self.algorithm!r}; valid algorithms: {', '.join(algorithm_names())}")
        for name in ("total_env_steps", "log_interval", "checkpoint_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
