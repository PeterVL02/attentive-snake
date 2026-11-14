"""The selectable training algorithms, by ``[train] algorithm`` (the contract is ``training.algorithm.Algorithm``)."""

from attentive_snake.training.algorithm import Algorithm
from attentive_snake.training.dqn import ALGORITHM as DQN
from attentive_snake.training.ppo import ALGORITHM as PPO

_ALGORITHMS: dict[str, Algorithm] = {algorithm.name: algorithm for algorithm in (PPO, DQN)}


def algorithm_names() -> tuple[str, ...]:
    return tuple(_ALGORITHMS)


def get_algorithm(name: str) -> Algorithm:
    """The algorithm called ``name`` (``ValueError`` naming the valid ones otherwise)."""
    if name not in _ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; valid algorithms: {', '.join(_ALGORITHMS)}")
    return _ALGORITHMS[name]
