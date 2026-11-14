"""What followed each env's action in a vector step, shared by the algorithms that learn from transitions."""

import numpy as np

from attentive_snake.env.types import VecEnvStep


def next_observations(step: VecEnvStep) -> np.ndarray:
    """What followed each env's action: ``step.observations`` with every finished env's final observation put back.

    Auto-reset already replaced a finished env's observation by the first one of its next
    episode; that observation must never become the next state of the last transition.
    """
    following = np.array(step.observations, dtype=np.float32, copy=True)
    for index, final in step.final_observations.items():
        following[index] = final
    return following
