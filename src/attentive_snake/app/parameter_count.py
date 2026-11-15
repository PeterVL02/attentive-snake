"""Parameter count of a policy: of a built one, of an experiment config, of a run's config file (cached)."""

from functools import lru_cache
from pathlib import Path

from torch import nn

from attentive_snake.config import ExperimentConfig, load_experiment
from attentive_snake.factory import make_env, make_policy


def count_parameters(policy: nn.Module) -> int:
    return sum(parameter.numel() for parameter in policy.parameters())


def config_parameter_count(config: ExperimentConfig) -> int:
    """Trainable parameters of the policy ``config`` describes (built on the CPU, nothing is written)."""
    return count_parameters(make_policy(config, make_env(config)))


def run_parameter_count(config_file: Path) -> int | None:
    """Same for a run's config file, cached per file version; ``None`` if it cannot be built."""
    try:
        return _count(str(config_file), config_file.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError):
        return None


@lru_cache(maxsize=64)
def _count(config_file: str, _mtime_ns: int) -> int:
    return config_parameter_count(load_experiment(Path(config_file)))
