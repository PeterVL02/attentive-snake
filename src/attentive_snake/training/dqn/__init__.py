"""DQN (deep Q-network, Double DQN by default): off-policy, value-based training of a Q-value head from a replay buffer.

Optional, each behind its own setting: n-step targets (``n_step.py``), a dueling head (``head.py`` picks it),
prioritized replay (``prioritized_replay.py``), a squared-error loss (``loss.py``).
"""

from attentive_snake.training.algorithm import Algorithm, TrainingMetric
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.dqn.head import head_name
from attentive_snake.training.dqn.learner import DqnLearner

METRICS = (
    TrainingMetric("loss", "TD loss", "Mean loss of the gradient steps since the previous record: the error between Q(s, a) and its target (the rewards of the next n_step steps plus the target network's bootstrap), penalised as dqn.loss says (Huber by default)."),
    TrainingMetric("q_mean", "mean Q-value", "Mean Q-value of the sampled actions: what the network expects to collect from there (discounted, in reward units)."),
    TrainingMetric("td_error", "mean |TD error|", "Mean absolute difference between Q(s, a) and its target in the sampled batches (with n_step above 1 the target looks that many steps ahead)."),
    TrainingMetric("epsilon", "epsilon", "Probability of a random action while collecting experience (falls over epsilon_fraction of training)."),
    TrainingMetric("priority_beta", "priority beta", "The importance-sampling exponent (how much of the sampling bias the loss weights correct); rises from priority_beta_start to 1 over training.", "dqn.prioritized"),
    TrainingMetric("priority_mean", "mean priority", "Mean sampling priority over the stored transitions, (|TD error| + epsilon) ** alpha; falls as the network predicts better.", "dqn.prioritized"),
    TrainingMetric("priority_max", "max priority", "The largest priority set so far, given to every new transition so it is replayed at least once.", "dqn.prioritized"),
    TrainingMetric("replay_size", "replay buffer size", "Transitions in the replay buffer (up to buffer_size)."),
    TrainingMetric("gradient_steps", "gradient steps", "Gradient steps taken by this training process so far (restarts at 0 on resume)."),
)

ALGORITHM = Algorithm(
    name="dqn",
    label="DQN",
    description=(
        "Deep Q-network (Double DQN by default): off-policy and value-based. The network outputs one Q-value per action; "
        "it plays epsilon-greedily, stores what happened in a replay buffer and learns from random batches of it "
        "against a slowly updated target network. Plays greedily (highest Q-value) when evaluated or watched. "
        "Optional: n-step targets (n_step), a dueling head (dueling), prioritized replay (prioritized), a squared-error loss (loss)."
    ),
    settings_type=DqnConfig,
    head=head_name,
    metrics=METRICS,
    build=DqnLearner,
    locked_settings={"dueling": "the parent's weights only fit the same head (one Q-value output, or the value and advantage streams)"},
)

__all__ = ["ALGORITHM", "DqnConfig"]
