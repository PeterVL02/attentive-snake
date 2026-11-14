"""With ``train.bootstrap_truncated`` off, training is bit-identical to the code before the flag existed.

``tests/data/flag_off_reference_weights.pt`` holds the weights of ``short_training()`` recorded on
``main`` before truncation bootstrapping was added (commit 778ec73), on CPU. The run uses Snake with a
starvation limit, so truncated episodes do occur, and a history of two frames. If a torch upgrade
changes the numbers, re-record on a checkout of that commit:
``python -c "import runpy; runpy.run_path('tests/test_flag_off_reference.py')['record']()"``.
"""

from pathlib import Path

import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.training import NullLogger, build_trainer
from attentive_snake.training.runtime import seed_everything

REFERENCE = Path(__file__).resolve().parent / "data" / "flag_off_reference_weights.pt"

TABLES = {
    "game": {"name": "snake", "width": 5, "height": 5, "max_steps_without_food": 5},
    "model": {"kind": "mlp", "d_model": 16, "num_layers": 1},
    "train": {
        "total_env_steps": 3 * 4 * 32,
        "device": "cpu",
        "seed": 3,
        "log_interval": 1,
    },
    "ppo": {"rollout_length": 32, "num_minibatches": 2},
    "env": {"num_envs": 4, "history_length": 2},
}


def short_training(tables: dict = TABLES) -> tuple[dict[str, torch.Tensor], int]:
    """Final weights and the number of truncated steps of a three-update Snake run (``TABLES``: no ``bootstrap_truncated`` key)."""
    config = experiment_from_tables(tables)
    env = make_vec_env(config)
    truncations = []
    step = env.step

    def counting_step(actions):
        result = step(actions)
        truncations.append(int(result.truncated.sum()))
        return result

    env.step = counting_step
    seed_everything(config.train.seed)
    policy = make_policy(config, env)
    build_trainer(policy, env, config.train, config.algorithm_settings, NullLogger()).train()
    return {name: value.detach().clone() for name, value in policy.state_dict().items()}, sum(truncations)


def record() -> None:
    weights, _ = short_training()
    REFERENCE.parent.mkdir(exist_ok=True)
    torch.save(weights, REFERENCE)


def test_flag_off_training_is_bit_identical_to_before_the_flag() -> None:
    expected = torch.load(REFERENCE)
    weights, truncations = short_training()
    assert truncations > 0  # the run exercises truncated episodes
    assert weights.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(weights[name], value), name
