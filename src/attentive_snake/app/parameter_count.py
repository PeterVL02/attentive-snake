"""Parameter count of a run's policy, cached per config file version (building a policy is not free)."""

from functools import lru_cache
from pathlib import Path

from attentive_snake.config import load_experiment
from attentive_snake.factory import make_env, make_policy


def run_parameter_count(config_file: Path) -> int | None:
    """Trainable parameters of the policy the config describes; ``None`` if it cannot be built."""
    try:
        return _count(str(config_file), config_file.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError):
        return None


@lru_cache(maxsize=64)
def _count(config_file: str, _mtime_ns: int) -> int:
    config = load_experiment(Path(config_file))
    policy = make_policy(config, make_env(config))
    return sum(parameter.numel() for parameter in policy.parameters())
