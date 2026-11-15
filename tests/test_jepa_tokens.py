"""The jepa_tokens kind: jepa's heads and objective on the hybrid trunk's whole token set, with a per-token predictor."""

import dataclasses

import pytest
import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.auxiliary import TransitionBatch, auxiliary_objective
from attentive_snake.models.blocks.token_predictor import TokenPredictor
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, build_policy
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.models.value_training import trained_values
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger
from attentive_snake.training.auxiliary import STATE_KEY
from attentive_snake.viz import diagram

A = 4
CONFIG = ModelConfig(kind="jepa_tokens", d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=1, prediction_coef=1.0)
SPECS = {"grid": ObservationSpec("grid", (3, 5, 5), 1), "vector": ObservationSpec("vector", (6,), 1)}
TOKENS = {"grid": 1 + 25, "vector": 1 + 6}  # class token + one per cell / number
TRUNK = {"grid": ["stem", "project", "tokens", "positional"], "vector": ["stem", "project"]}
TAIL = ["cls", "blocks", "norm", "predictor"]  # no pool: the decoders read the class token of a token set
ENCODER = ["cls", "blocks", "norm"]


def make(observation="grid", head="q_values", **changes):
    torch.manual_seed(0)
    return build_policy(dataclasses.replace(CONFIG, **changes), SPECS[observation], A, head)


def batch(policy, size=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = policy.observation_shape
    return TransitionBatch(
        torch.rand(size, *shape, generator=generator),
        torch.randint(0, A, (size,), generator=generator),
        torch.rand(size, *shape, generator=generator),
        torch.ones(size),
    )


@pytest.mark.parametrize("observation", sorted(SPECS))
def test_both_heads_act_on_the_class_token_of_the_predicted_token_sets(observation):
    q, ac = make(observation, "q_values"), make(observation, "actor_critic")
    obs = torch.rand(5, *SPECS[observation].shape)
    out = q(obs)
    assert isinstance(out, QValues) and out.q_values.shape == (5, A)
    tokens = q.encode(obs)
    assert tokens.shape == (5, TOKENS[observation], 16)
    predicted = q.predictor(tokens)
    assert predicted.shape == (5, A, TOKENS[observation], 16)
    assert torch.allclose(out.q_values, q.net["value_decoder"](predicted[:, :, 0]))
    policy = ac(obs)
    assert isinstance(policy, PolicyOutput) and policy.logits.shape == (5, A) and policy.value.shape == (5,)
    tokens = ac.encode(obs)
    assert torch.allclose(policy.value, ac.net["value_decoder"](tokens[:, 0]))
    assert torch.allclose(policy.logits, ac.net["policy_decoder"](ac.predictor(tokens)[:, :, 0]))
    assert list(q.net) == [*TRUNK[observation], *TAIL, "value_decoder"]
    assert list(ac.net) == [*TRUNK[observation], *TAIL, "policy_decoder", "value_decoder"]


def test_the_predictor_reads_token_sets_only_and_chains_on_its_own_predictions():
    policy = make()
    tokens = policy.encode(torch.rand(3, 3, 5, 5))
    actions = torch.tensor([0, 2, 3])
    once = policy.predictor.step(tokens, actions)
    twice = policy.predictor.step(once, actions)
    assert once.shape == twice.shape == tokens.shape and not torch.allclose(once, twice)
    every = policy.predictor(tokens)
    assert torch.allclose(every[torch.arange(3), actions], once, atol=1e-6)  # forward = step for every action


def test_predictor_layers_sets_the_predictor_depth_and_num_layers_the_encoder():
    policy = make(num_layers=3, predictor_layers=2)
    assert isinstance(policy.predictor, TokenPredictor)
    assert len(policy.predictor.blocks) == 2 and len(policy.net["blocks"]) == 3
    assert len(make().predictor.blocks) == 1  # the default


@pytest.mark.parametrize("observation", sorted(SPECS))
def test_attention_maps_are_the_encoders_and_the_diagram_renders(observation):
    policy = make(observation)
    maps = policy.attention_maps(torch.rand(3, *SPECS[observation].shape))
    assert policy.token_layout.num_tokens == TOKENS[observation]
    assert len(maps) == CONFIG.num_layers
    assert all(m.shape == (3, 2, TOKENS[observation], TOKENS[observation]) for m in maps)
    text = diagram(policy, "text")
    assert "TransformerBlock" in text and "TokenPredictor" in text


def test_the_embedding_encoder_is_the_whole_observation_to_tokens_path_and_shares_the_weights():
    policy = make("grid")
    encoder = policy.embedding_encoder()
    obs = torch.rand(4, 3, 5, 5)
    assert torch.equal(encoder(obs), policy.encode(obs))
    assert {id(p) for p in encoder.parameters()} <= {id(p) for p in policy.parameters()}
    trunk = sum(p.numel() for name in [*TRUNK["grid"], *ENCODER] for p in policy.net[name].parameters())
    assert sum(p.numel() for p in encoder.parameters()) == trunk
    assert not any(key.startswith(("front", "tail")) for key in policy.state_dict())


def test_no_objective_without_the_coefficient():
    assert auxiliary_objective(make(prediction_coef=0.0)) is None


@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_the_prediction_loss_learns_and_never_trains_the_target_encoder(value_training):
    policy = make(value_training=value_training)
    objective = auxiliary_objective(policy)
    target_parameters = list(objective.target.module.parameters())
    assert target_parameters and all(not p.requires_grad for p in target_parameters)
    assert not {id(p) for p in target_parameters} & {id(p) for p in policy.parameters()}
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    data = batch(policy)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        terms = objective.loss(data)
        terms.loss.backward()
        optimizer.step()
        objective.after_step()
        losses.append(terms.metrics["prediction_loss"])
    assert losses[-1] < 0.5 * losses[0]
    assert set(terms.metrics) == {metric.key for metric in KIND_INFO["jepa_tokens"].metrics}


def test_the_prediction_loss_compares_every_token():
    policy = make()
    objective = auxiliary_objective(policy)
    data = batch(policy)
    with torch.no_grad():
        predicted = policy.predictor.step(policy.encode(data.observations), data.actions)
        target = objective.target(data.next_observations)
        expected = (1 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1)).mean()
    assert objective.loss(data).metrics["prediction_loss"] == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize("trains_encoder", [True, False])
def test_the_prediction_loss_reaches_the_encoder_only_when_asked(trains_encoder):
    policy = make(prediction_trains_encoder=trains_encoder)
    auxiliary_objective(policy).loss(batch(policy)).loss.backward()
    encoder_grad = sum(float(p.grad.abs().sum()) for name in [*TRUNK["grid"], *ENCODER] for p in policy.net[name].parameters() if p.grad is not None)
    assert (encoder_grad > 0) == trains_encoder
    assert policy.predictor.out.weight.grad.abs().sum() > 0


def test_actual_trains_the_value_of_the_real_next_class_token_and_never_the_predictor():
    policy = make(value_training="actual")
    data = batch(policy)
    values = trained_values(policy, data.observations, data.actions, data.next_observations)
    assert torch.allclose(values, policy.net["value_decoder"](policy.encode(data.next_observations)[:, 0]))
    values.sum().backward()
    assert all(p.grad is None for p in policy.predictor.parameters())
    assert trained_values(make(), data.observations, data.actions, data.next_observations) is None


def tiny_dqn(model: dict, steps: int = 96, **dqn):
    return experiment_from_tables({
        "game": {"width": 4, "height": 4},
        "model": {"kind": "jepa_tokens", "d_model": 8, "num_heads": 2, "num_layers": 1, "conv_channels": 4, "conv_layers": 1, "prediction_coef": 1.0, **model},
        "train": {"algorithm": "dqn", "total_env_steps": steps, "device": "cpu", "log_interval": 32},
        "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8, **dqn},
        "env": {"num_envs": 2},
    })


@pytest.mark.parametrize("prioritized", [False, True])
@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_dqn_trains_the_objective(value_training, prioritized):
    config = tiny_dqn({"value_training": value_training}, prioritized=prioritized)
    env = make_vec_env(config)
    policy = make_policy(config, env)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    trainer = make_trainer(config, policy, env, NullLogger())
    records = trainer.train()
    assert {"prediction_loss", "embedding_std", "value_gap", "td_error"} <= set(records[-1])
    if prioritized:
        assert {"priority_beta", "priority_mean", "priority_max"} <= set(records[-1])
    assert any(not torch.equal(before[k], v) for k, v in policy.state_dict().items() if k.startswith("net.predictor."))
    state = trainer.learner.extra_state()
    assert set(state) == {"target", STATE_KEY} and "target_encoder" in state[STATE_KEY]


def test_the_target_encoder_is_checkpointed_and_restored_on_resume(tmp_path):
    config = tiny_dqn({})
    config = dataclasses.replace(config, train=dataclasses.replace(config.train, checkpoint_dir=tmp_path))
    env = make_vec_env(config)
    trainer = make_trainer(config, make_policy(config, env), env, NullLogger())
    trainer.train()
    checkpoint = trainer.save_checkpoint()
    saved = trainer.learner._auxiliary._objective.target.state_dict()
    online = trainer.policy.embedding_encoder().state_dict()
    assert set(saved) == set(online) and any(not torch.equal(saved[k], online[k]) for k in saved)  # the EMA lags the encoder

    env = make_vec_env(config)
    resumed = make_trainer(config, make_policy(config, env), env, NullLogger())
    resumed.restore(checkpoint, keep_step_count=True)
    restored = resumed.learner._auxiliary._objective.target.state_dict()
    assert all(torch.equal(restored[k], saved[k]) for k in saved)


def test_ppo_trains_the_prediction_loss():
    config = experiment_from_tables({
        "game": {"width": 4, "height": 4, "max_steps_without_food": 8},
        "model": {"kind": "jepa_tokens", "d_model": 8, "num_heads": 2, "num_layers": 1, "conv_channels": 4, "conv_layers": 1, "prediction_coef": 1.0},
        "train": {"total_env_steps": 64, "device": "cpu", "log_interval": 32},
        "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
        "env": {"num_envs": 4},
    })
    env = make_vec_env(config)
    trainer = make_trainer(config, make_policy(config, env), env, NullLogger())
    records = trainer.train()
    assert {"prediction_loss", "embedding_std"} <= set(records[-1]) and set(trainer.learner.extra_state()) == {STATE_KEY}
