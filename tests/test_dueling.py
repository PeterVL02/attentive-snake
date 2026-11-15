"""The dueling head: Q = V + A - mean(A), for every model kind, chosen by ``dqn.dueling``, locked on fork."""

import dataclasses

import pytest
import torch

from attentive_snake.config import ExperimentConfig
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.fork_rules import locked_field_changes, locked_fields, require_forkable
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.acting import action_readout, greedy_actions
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.dueling import dueling_q_values
from attentive_snake.models.registry import POLICY_KINDS, build_policy
from attentive_snake.models.types import QValues
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger
from attentive_snake.training.algorithms import get_algorithm
from attentive_snake.training.config import TrainConfig
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.viz import diagram

A = 4
SMALL = dict(d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2)
SPECS = {
    "grid": ObservationSpec("grid", (2 * (3 + A), 5, 6), 2, A),
    "vector": ObservationSpec("vector", (2 * (5 + A),), 2, A),
}
JEPA_KINDS = [kind for kind in POLICY_KINDS if kind.startswith("jepa")]


def make(kind: str, observation: str, head: str, **model):
    torch.manual_seed(0)
    return build_policy(ModelConfig(kind=kind, **SMALL, **model), SPECS[observation], A, head)


def test_the_combination_by_hand() -> None:
    q = dueling_q_values(torch.tensor([10.0, -1.0]), torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 6.0]]))
    assert q.tolist() == [[9.0, 10.0, 11.0], [-3.0, -3.0, 3.0]]  # V + A - mean(A); mean A = 2 in both rows
    assert q.mean(dim=1).tolist() == [10.0, -1.0]  # the mean Q-value is the state value


@pytest.mark.parametrize("observation", sorted(SPECS))
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_every_kind_builds_with_the_dueling_head_on_the_same_trunk(kind: str, observation: str) -> None:
    dueling, plain = make(kind, observation, "dueling"), make(kind, observation, "q_values")
    obs = torch.rand(3, *SPECS[observation].shape)
    out = dueling(obs)
    assert isinstance(out, QValues) and out.q_values.shape == (3, A)
    assert greedy_actions(out).shape == (3,) and action_readout(out).kind == "q_values"  # acting needs no change

    forms = type(dueling).heads
    names, plain_names = list(forms["dueling"].modules(16, A)), list(forms["q_values"].modules(16, A))
    assert list(dueling.net)[-len(names):] == names and dueling.head_name == "dueling"
    trunk = lambda policy, head: {key for key in policy.state_dict() if key.split(".")[1] not in head}
    assert trunk(dueling, names) == trunk(plain, plain_names)  # only the head differs
    assert set(dueling.state_dict()) != set(plain.state_dict())  # so a checkpoint of one does not fit the other
    assert type(dueling.net[names[1]]).__name__ in diagram(dueling)  # the advantage part is drawn

    # The identity: the mean Q-value over the actions is the value part, the advantages only spread around it.
    value_name, advantage_name = names
    with torch.no_grad():
        dueling.net[advantage_name].linear.weight.zero_()
        bias = dueling.net[advantage_name].linear.bias  # one per action, or one for every predicted embedding (jepa)
        bias.copy_(torch.tensor([1.0, 2.0, 3.0, 6.0])[: bias.numel()])
        flat = dueling(obs).q_values
    if kind in JEPA_KINDS:  # one advantage decoder for every predicted embedding: equal advantages cancel entirely
        assert torch.allclose(flat, flat[:, :1].expand(-1, A), atol=1e-6)
    else:  # advantages [1, 2, 3, 6] (mean 3): Q - V = [-2, -1, 0, 3] in every state
        assert torch.allclose(flat - flat.mean(dim=1, keepdim=True), torch.tensor([-2.0, -1.0, 0.0, 3.0]).expand(3, A), atol=1e-5)
    with torch.no_grad():
        mean_before = dueling(obs).q_values.mean(dim=1)
        dueling.net[value_name].linear.bias.add_(7.0)
        assert torch.allclose(dueling(obs).q_values.mean(dim=1), mean_before + 7.0, atol=1e-5)  # the value part is the mean Q


@pytest.mark.parametrize("kind", JEPA_KINDS)
def test_jepa_kinds_refuse_the_dueling_head_with_actual_value_training(kind: str) -> None:
    with pytest.raises(ValueError, match="cannot combine the dueling head .* with value_training = 'actual'"):
        make(kind, "grid", "dueling", value_training="actual", prediction_coef=1.0)
    make(kind, "grid", "q_values", value_training="actual", prediction_coef=1.0)  # still fine without it
    make(kind, "grid", "dueling", value_training="predicted", prediction_coef=1.0)


def test_existing_head_modules_keep_their_names() -> None:
    assert [key for key in make("mlp", "vector", "q_values").state_dict() if "head" in key] == ["net.q_head.linear.weight", "net.q_head.linear.bias"]
    assert [key.split(".")[1] for key in make("mlp", "vector", "dueling").state_dict()][-8:] == ["dueling_value"] * 4 + ["dueling_advantage"] * 4
    assert list(type(make("jepa", "vector", "q_values")).heads["q_values"].modules(16, A)) == ["value_decoder"]


def tables(**dqn) -> dict:
    return {
        "game": {"width": 4, "height": 4},
        "model": {"kind": "mlp", "d_model": 8, "num_layers": 1},
        "train": {"algorithm": "dqn", "total_env_steps": 64, "device": "cpu", "log_interval": 1},
        "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8} | dqn,
        "env": {"num_envs": 2},
    }


def test_the_dqn_setting_chooses_the_head_and_the_run_trains() -> None:
    dqn = get_algorithm("dqn")
    assert dqn.head(DqnConfig()) == "q_values" and dqn.head(DqnConfig(dueling=True)) == "dueling"
    assert get_algorithm("ppo").head(None) == "actor_critic"
    config = experiment_from_tables(tables(dueling=True, n_step=3, loss="mse"))
    env = make_vec_env(config)
    policy = make_policy(config, env)
    assert policy.head_name == "dueling" and make_policy(experiment_from_tables(tables()), env).head_name == "q_values"
    records = make_trainer(config, policy, env, NullLogger()).train()
    assert any("loss" in record for record in records)


def test_dueling_is_locked_on_fork_and_the_other_new_settings_are_not() -> None:
    assert "dqn.dueling" in locked_fields("snake") and "dqn.dueling" in locked_fields("lunar_lander")
    assert not {"dqn.n_step", "dqn.loss"} & set(locked_fields("snake"))
    parent = ExperimentConfig(train=TrainConfig(algorithm="dqn"))
    child = dataclasses.replace(parent, algorithm_settings=DqnConfig(dueling=True))
    assert locked_field_changes(parent, child) == ["dqn.dueling"]
    with pytest.raises(ValueError, match=r"dqn\.dueling \(False -> True\): the parent's weights only fit the same head"):
        require_forkable(parent, child)
    free = dataclasses.replace(parent, algorithm_settings=DqnConfig(n_step=3, loss="mse"))
    assert locked_field_changes(parent, free) == []
    assert locked_field_changes(ExperimentConfig(), ExperimentConfig()) == []  # a PPO run has no [dqn] to compare
    assert locked_field_changes(ExperimentConfig(), child) == ["train.algorithm"]
