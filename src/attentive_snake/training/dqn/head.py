"""Which policy head a DQN run trains (``models.heads``), from its settings."""

from attentive_snake.training.dqn.config import DqnConfig


def head_name(settings: DqnConfig) -> str:
    """``"dueling"`` with ``dqn.dueling``, else ``"q_values"``; both output ``QValues``."""
    return "dueling" if settings.dueling else "q_values"
