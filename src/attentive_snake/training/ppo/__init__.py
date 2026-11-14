"""PPO (proximal policy optimisation): on-policy actor-critic training of a policy-and-value head."""

from attentive_snake.training.algorithm import Algorithm, TrainingMetric
from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.training.ppo.learner import PpoLearner

METRICS = (
    TrainingMetric("policy_loss", "policy loss", "PPO's clipped-surrogate loss of the policy head (lower is not necessarily better)."),
    TrainingMetric("value_loss", "value loss", "Squared error of the value head against the GAE returns."),
    TrainingMetric("entropy", "entropy", "Randomness of the policy; falls as the policy commits to actions."),
    TrainingMetric("approx_kl", "approx. KL", "How far one update moved the policy (approximate KL divergence)."),
    TrainingMetric("clip_fraction", "clip fraction", "Share of samples whose probability ratio was clipped."),
    TrainingMetric("explained_variance", "explained variance", "How much of the returns' variance the value head explains (1 = perfect)."),
    TrainingMetric("epochs_completed", "epochs completed", "Epochs run in the latest update (fewer than num_epochs when target_kl stopped it)."),
)

ALGORITHM = Algorithm(
    name="ppo",
    label="PPO",
    description=(
        "Proximal policy optimisation: on-policy actor-critic. The network has a policy head (action probabilities) "
        "and a value head; it learns from rollouts it has just played, a few epochs each, and then discards them."
    ),
    settings_type=PpoConfig,
    head="actor_critic",
    metrics=METRICS,
    build=PpoLearner,
)

__all__ = ["ALGORITHM", "PpoConfig"]
