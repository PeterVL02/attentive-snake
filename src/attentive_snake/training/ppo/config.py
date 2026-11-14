"""PPO's own settings (the ``[ppo]`` section). Pure data, no behaviour beyond validation."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PpoConfig:
    """Settings of PPO; one update consumes ``rollout_length * num_envs`` env steps. Descriptions in ``metadata["help"]``."""

    rollout_length: int = field(default=128, metadata={"help": "How many steps each environment plays before the network is updated. Longer rollouts give the learner more context per update but make updates less frequent."})
    num_epochs: int = field(default=4, metadata={"help": "How many times PPO re-uses each collected batch of experience. More epochs squeeze more out of the data but risk overfitting the batch."})
    num_minibatches: int = field(default=4, metadata={"help": "Each epoch splits the batch into this many pieces and takes one gradient step per piece. More pieces means smaller, noisier steps."})
    gae_lambda: float = field(default=0.95, metadata={"help": "Smoothing in the advantage estimate (GAE). Near 1 uses long real returns (less bias, more noise); near 0 relies on the value network's guesses (more bias, less noise)."})
    clip_coef: float = field(default=0.2, metadata={"help": "PPO's trust-region size: how far the new policy may move from the old one in a single update. Smaller is safer and slower; larger is faster and can destabilise training."})
    value_coef: float = field(default=0.5, metadata={"help": "Weight of the value-prediction loss relative to the policy loss. Raise it to make the network focus on predicting returns accurately."})
    entropy_coef: float = field(default=0.01, metadata={"help": "Weight of a bonus for keeping the policy random. Higher means more exploration and slower commitment to one behaviour; 0 lets the policy become deterministic quickly."})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Gradients are scaled down if their overall size exceeds this value, which protects against occasional huge updates."})
    normalize_advantages: bool = field(default=True, metadata={"help": "Rescale advantages to zero mean and unit variance within each minibatch, so the update size does not depend on the reward scale."})
    target_kl: float | None = field(default=None, metadata={"help": "Optional early stop: if the policy changes more than this (approximate KL divergence) during an update, the remaining epochs are skipped. Empty means never stop early."})

    def __post_init__(self) -> None:
        for name in ("rollout_length", "num_epochs", "num_minibatches"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        for name in ("clip_coef", "max_grad_norm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("Loss coefficients must be non-negative.")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive or None.")
