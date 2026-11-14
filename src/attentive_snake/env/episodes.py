"""Per-environment episode bookkeeping: return and length so far, and the summary when an episode ends."""

from attentive_snake.env.types import EnvStep, EpisodeSummary


class EpisodeTracker:
    """Accumulates each env's return and length; emits an ``EpisodeSummary`` when an episode ends."""

    def __init__(self, num_envs: int) -> None:
        self._returns = [0.0] * num_envs
        self._lengths = [0] * num_envs

    def reset(self) -> None:
        self._returns = [0.0] * len(self._returns)
        self._lengths = [0] * len(self._lengths)

    def update(self, env_index: int, step: EnvStep) -> EpisodeSummary | None:
        """Add a step; return a summary (and start counting afresh) if it ended the episode."""
        self._returns[env_index] += step.reward
        self._lengths[env_index] += 1
        if not (step.terminated or step.truncated):
            return None
        summary = EpisodeSummary(
            env_index=env_index,
            score=float(step.score),
            length=self._lengths[env_index],
            total_reward=self._returns[env_index],
            outcome=step.event,
        )
        self._returns[env_index], self._lengths[env_index] = 0.0, 0
        return summary
