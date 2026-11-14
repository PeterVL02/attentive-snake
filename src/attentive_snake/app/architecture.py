"""The policy network an experiment config describes, as diagrams and a parameter count."""

from attentive_snake.config import ExperimentConfig
from attentive_snake.factory import make_env, make_policy
from attentive_snake.viz import diagram


def describe_architecture(config: ExperimentConfig) -> dict:
    """Build env and policy on the CPU and describe the policy: ``{kind, algorithm, head, text, mermaid, parameters}``.

    The head is the one ``train.algorithm`` trains (``models.heads``), so the same kind shows
    a policy and value head for PPO and a Q-value head for DQN.
    """
    env = make_env(config)
    policy = make_policy(config, env)
    return {
        "game": config.game_name,
        "kind": config.model.kind,
        "algorithm": config.train.algorithm,
        "head": policy.head_name,
        "observation_shape": list(env.observation_shape),
        "text": diagram(policy, fmt="text"),
        "mermaid": diagram(policy, fmt="mermaid"),
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
    }
