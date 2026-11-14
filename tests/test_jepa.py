"""The jepa kind: its forms, its prediction objective (EMA target encoder), and how DQN trains it."""

import copy
import dataclasses

import pytest
import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.auxiliary import TransitionBatch, auxiliary_objective
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.jepa.ema import EmaCopy
from attentive_snake.models.jepa.objective import cosine_distance, masked_mean, squared_distance
from attentive_snake.models.registry import KIND_INFO, build_policy
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger
from attentive_snake.training.auxiliary import STATE_KEY
from attentive_snake.training.dqn.replay import ReplaySample
from attentive_snake.training.dqn.target import TargetNetwork
from attentive_snake.training.dqn.update import dqn_gradient_step

A = 4
CONFIG = ModelConfig(kind="jepa", d_model=16, num_layers=2, conv_channels=8, conv_layers=2)
SPECS = {
    ("grid", 1): ObservationSpec("grid", (3, 5, 5), 1),
    ("grid", 3): ObservationSpec("grid", (3 * (3 + A), 5, 5), 3, A),
    ("vector", 1): ObservationSpec("vector", (6,), 1),
    ("vector", 3): ObservationSpec("vector", (3 * (6 + A),), 3, A),
}


def make(spec_key=("vector", 1), head="q_values", **changes):
    torch.manual_seed(0)
    return build_policy(dataclasses.replace(CONFIG, **changes), SPECS[spec_key], A, head)


def batch(policy, size=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = policy.observation_shape
    return TransitionBatch(
        torch.rand(size, *shape, generator=generator),
        torch.randint(0, A, (size,), generator=generator),
        torch.rand(size, *shape, generator=generator),
        torch.ones(size),
    )


@pytest.mark.parametrize("spec_key", sorted(SPECS))
def test_both_forms_have_the_right_shapes_and_q_is_the_decoded_prediction(spec_key):
    q, ac = make(spec_key, "q_values"), make(spec_key, "actor_critic")
    obs = torch.rand(5, *SPECS[spec_key].shape)
    out = q(obs)
    assert isinstance(out, QValues) and out.q_values.shape == (5, A)
    z = q.encode(obs)
    predictions = q.predictor(z)
    assert z.shape == (5, 16) and predictions.shape == (5, A, 16)
    assert torch.allclose(out.q_values, q.net["value_decoder"](predictions))
    for action in range(A):  # the batched pass equals one action at a time
        one = q.predictor.step(z, torch.full((5,), action))
        assert torch.allclose(one, predictions[:, action], atol=1e-6)
    policy = ac(obs)
    assert isinstance(policy, PolicyOutput) and policy.logits.shape == (5, A) and policy.value.shape == (5,)
    assert torch.allclose(policy.value, ac.net["value_decoder"](ac.encode(obs)))
    assert list(q.net) == ["encoder", "readout", "predictor", "value_decoder"]
    assert list(ac.net) == ["encoder", "readout", "predictor", "policy_decoder", "value_decoder"]


def test_the_predictor_runs_on_its_own_predictions_for_lookahead():
    policy = make()
    z = policy.encode(torch.rand(2, 6))
    two_steps = policy.predictor(policy.predictor(z).flatten(0, 1))  # every action, then every action again
    assert two_steps.shape == (2 * A, A, 16)
    values = policy.net["value_decoder"](two_steps).view(2, A, A)
    assert values.shape == (2, A, A)


def test_ema_update_math():
    online = torch.nn.Linear(3, 2)
    ema = EmaCopy(online, decay=0.75)
    before = copy.deepcopy(ema.module.state_dict())
    with torch.no_grad():
        online.weight.add_(1.0)
    ema.update()
    expected = 0.75 * before["weight"] + 0.25 * online.weight
    assert torch.allclose(ema.module.weight, expected)
    assert not any(p.requires_grad for p in ema.module.parameters())
    ema.reset()
    assert torch.equal(ema.module.weight, online.weight)
    assert torch.equal(EmaCopy(online, 0.0).module.weight, online.weight)


def test_distances_and_the_terminal_mask():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert torch.allclose(cosine_distance(a, torch.tensor([[2.0, 0.0], [1.0, 0.0]])), torch.tensor([0.0, 1.0]))
    assert torch.allclose(squared_distance(a, torch.zeros(2, 2)), torch.tensor([0.5, 2.0]))
    assert masked_mean(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0])) == 3.0
    assert masked_mean(torch.tensor([1.0, 3.0]), torch.zeros(2)) == 0.0


@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_the_prediction_loss_learns_and_never_trains_the_target_encoder(value_training):
    policy = make(value_training=value_training)
    objective = auxiliary_objective(policy)
    target_parameters = list(objective.target.module.parameters())
    assert all(not p.requires_grad for p in target_parameters)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    assert not {id(p) for p in target_parameters} & {id(p) for group in optimizer.param_groups for p in group["params"]}
    data = batch(policy, size=16)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        terms = objective.loss(data)
        terms.loss.backward()
        assert all(p.grad is None for p in target_parameters)
        optimizer.step()
        objective.after_step()
        losses.append(terms.metrics["prediction_loss"])
    assert losses[-1] < 0.5 * losses[0]
    assert set(terms.metrics) == {key.key for key in KIND_INFO["jepa"].metrics}


def test_no_objective_without_the_coefficient_and_actual_needs_one():
    assert auxiliary_objective(make(prediction_coef=0.0)) is None
    assert auxiliary_objective(build_policy(ModelConfig(kind="mlp"), SPECS["vector", 1], A, "q_values")) is None
    with pytest.raises(ValueError, match="needs prediction_coef > 0"):
        ModelConfig(kind="jepa", value_training="actual", prediction_coef=0.0)
    with pytest.raises(ValueError, match="value_training must be one of"):
        ModelConfig(value_training="both")


def _sample(policy, size=8):
    data = batch(policy, size)
    return ReplaySample(data.observations, data.actions, torch.rand(size), data.next_observations, torch.zeros(size), torch.ones(size), None)


@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_the_td_loss_reaches_the_predictor_only_when_it_trains_predicted_values(value_training):
    policy = make(value_training=value_training, prediction_coef=1.0)
    target = TargetNetwork(policy)
    assert set(target.state_dict()) == set(policy.state_dict())  # a copy of the policy: no EMA encoder inside
    sample = _sample(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
    dqn_gradient_step(policy, target, optimizer, sample, 0.9, True, 10.0)  # no auxiliary objective passed
    predictor_grad = sum(float(p.grad.abs().sum()) for p in policy.predictor.parameters() if p.grad is not None)
    decoder_grad = float(policy.net["value_decoder"].linear.weight.grad.abs().sum())
    assert decoder_grad > 0
    assert (predictor_grad > 0) == (value_training == "predicted")
    if value_training == "actual":  # Q(s, a) as acted on is unchanged: still the decoded prediction
        obs = sample.observations
        assert torch.allclose(policy(obs).q_values, policy.net["value_decoder"](policy.predictor(policy.encode(obs))))


def tiny_dqn(model: dict, steps: int = 96):
    return experiment_from_tables({
        "game": {"width": 4, "height": 4},
        "model": {"kind": "jepa", "d_model": 8, "num_layers": 1, "conv_channels": 4, "conv_layers": 1, **model},
        "train": {"algorithm": "dqn", "total_env_steps": steps, "device": "cpu", "log_interval": 32},
        "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8},
        "env": {"num_envs": 2},
    })


@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_dqn_logs_the_objective_and_checkpoints_its_target_encoder(value_training):
    config = tiny_dqn({"value_training": value_training})
    env = make_vec_env(config)
    trainer = make_trainer(config, make_policy(config, env), env, NullLogger())
    records = trainer.train()
    assert {"prediction_loss", "embedding_std"} <= set(records[-1])
    state = trainer.learner.extra_state()
    assert set(state) == {"target", STATE_KEY} and "target_encoder" in state[STATE_KEY]
    assert not any("target_encoder" in key for key in trainer.policy.state_dict())  # evaluation and watching need only the policy


def test_without_the_coefficient_dqn_has_no_objective_state_or_metrics():
    config = tiny_dqn({"prediction_coef": 0.0})
    env = make_vec_env(config)
    trainer = make_trainer(config, make_policy(config, env), env, NullLogger())
    records = trainer.train()
    assert not {"prediction_loss", "embedding_std"} & set().union(*records)
    assert set(trainer.learner.extra_state()) == {"target"}


def test_restoring_without_objective_state_restarts_the_target_from_the_loaded_encoder(tmp_path):
    parent = tiny_dqn({"prediction_coef": 0.0})
    env = make_vec_env(parent)
    trainer = make_trainer(dataclasses.replace(parent, train=dataclasses.replace(parent.train, checkpoint_dir=tmp_path)), make_policy(parent, env), env, NullLogger())
    trainer.train()
    checkpoint = trainer.save_checkpoint()

    child = tiny_dqn({"prediction_coef": 0.5})  # a fork may switch the objective on
    env = make_vec_env(child)
    forked = make_trainer(child, make_policy(child, env), env, NullLogger())
    forked.restore(checkpoint, keep_step_count=False)
    target = forked.learner._auxiliary._objective.target.module
    online = forked.policy.net
    assert torch.equal(target[0].stem[0].weight, online["encoder"].stem[0].weight)
    assert torch.equal(target[1].linear.weight, online["readout"].linear.weight)

    saved = forked.learner.extra_state()[STATE_KEY]
    resumed = make_trainer(child, make_policy(child, make_vec_env(child)), make_vec_env(child), NullLogger())
    resumed.learner.load_extra_state({"target": forked.learner.extra_state()["target"], STATE_KEY: saved})
    assert torch.equal(resumed.learner._auxiliary._objective.target.module[1].linear.weight, target[1].linear.weight)


def test_jepa_trains_resumes_and_forks_through_the_script(tmp_path):
    """DQN with value_training = actual through scripts/train.py; resume keeps the target encoder; a fork may change the objective."""
    from attentive_snake.run_dir import list_checkpoints
    from attentive_snake.training.checkpoint import read_checkpoint
    from attentive_snake.training.metrics import read_metrics
    from test_train_cli import TINY, train
    from test_train_new_kinds import DQN

    (tmp_path / "tiny.toml").write_text(TINY)
    run_dir = tmp_path / "run"
    kind = ["--set", "model.kind=jepa", "--set", "model.conv_channels=8", "--set", "model.conv_layers=1", "--set", "model.value_training=actual"]
    result = train("--run-dir", run_dir, "--config", tmp_path / "tiny.toml", *kind, *DQN)
    assert result.returncode == 0, result.stderr
    saved = read_checkpoint(list_checkpoints(run_dir)[-1])
    assert "target_encoder" in saved["learner"][STATE_KEY] and "net.predictor.action_embedding.weight" in saved["policy"]
    assert "prediction_loss" in read_metrics(run_dir / "metrics.jsonl")[-1]
    assert train("--resume", run_dir, "--steps", 96).returncode == 0
    forked = train("--run-dir", tmp_path / "child", "--fork", run_dir, "--set", "model.prediction_coef=0.5", "--set", "model.target_ema=0.9", "--steps", 32)
    assert forked.returncode == 0, forked.stderr
    refused = train("--run-dir", tmp_path / "other", "--fork", run_dir, "--set", "model.num_layers=2")
    assert refused.returncode != 0 and "model.num_layers" in refused.stderr
