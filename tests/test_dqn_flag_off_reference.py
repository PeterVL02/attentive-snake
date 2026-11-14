"""With ``dqn.prioritized`` off, DQN training is bit-identical to the code before prioritized replay existed.

``tests/data/dqn_flag_off_reference_weights.pt`` holds the final weights and the replay-sampling
fingerprint of ``short_training()`` recorded on ``main`` (commit 51dafee), on CPU. If a torch upgrade
changes the numbers, re-record on a checkout of that commit:
``python -c "import runpy; runpy.run_path('tests/test_dqn_flag_off_reference.py')['record']()"``.
"""

from pathlib import Path

import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.training import NullLogger, build_trainer
from attentive_snake.training.runtime import seed_everything

REFERENCE = Path(__file__).resolve().parent / "data" / "dqn_flag_off_reference_weights.pt"

TABLES = {
    "game": {"name": "snake", "width": 5, "height": 5, "max_steps_without_food": 20},
    "model": {"kind": "mlp", "d_model": 16, "num_layers": 1},
    "train": {"algorithm": "dqn", "total_env_steps": 400, "device": "cpu", "seed": 3, "log_interval": 1},
    "dqn": {"buffer_size": 120, "batch_size": 8, "learning_starts": 16, "target_update_interval": 40},
    "env": {"num_envs": 4},
}


def short_training() -> dict[str, torch.Tensor]:
    """Final weights of a tiny Snake DQN run (the buffer wraps around: 400 steps into 120 slots)."""
    config = experiment_from_tables(TABLES)
    env = make_vec_env(config)
    seed_everything(config.train.seed)
    policy = make_policy(config, env)
    build_trainer(policy, env, config.train, config.algorithm_settings, NullLogger()).train()
    return {name: value.detach().clone() for name, value in policy.state_dict().items()}


def record() -> None:
    torch.save(short_training(), REFERENCE)


def test_prioritized_off_dqn_training_is_bit_identical_to_before_prioritized_replay() -> None:
    expected = torch.load(REFERENCE)
    weights = short_training()
    assert weights.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(weights[name], value), name
