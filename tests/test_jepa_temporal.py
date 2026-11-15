"""The jepa_temporal kind: jepa's predictor, heads and objective on the temporal_transformer trunk (attention over time)."""

import copy
import dataclasses

import pytest
import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.auxiliary import TransitionBatch, auxiliary_objective
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, build_policy
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger
from attentive_snake.training.auxiliary import STATE_KEY
from attentive_snake.viz import diagram

A, K = 4, 3
CONFIG = ModelConfig(kind="jepa_temporal", d_model=16, num_heads=2, num_layers=2, prediction_coef=1.0)
BASE = {"grid": 3, "vector": 6}
POSITIONS = {"grid": 25, "vector": 6}


def spec(observation: str, actions: bool = True) -> ObservationSpec:
    channels = K * (BASE[observation] + (A if actions else 0))
    shape = (channels, 5, 5) if observation == "grid" else (channels,)
    return ObservationSpec(observation, shape, K, A if actions else 0)


def trunk(actions: bool = True) -> list[str]:
    """temporal_transformer's front: TOKEN_STAGES of one frame (the same names for grid and vector), then time and actions."""
    return ["embed", "positional", "temporal", *(["action_embed"] if actions else [])]


TAIL = ["cls", "blocks", "norm", "pool", "predictor"]


def make(observation="grid", head="q_values", actions=True, **changes):
    torch.manual_seed(0)
    return build_policy(dataclasses.replace(CONFIG, **changes), spec(observation, actions), A, head)


def batch(policy, size=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = policy.observation_shape
    return TransitionBatch(
        torch.rand(size, *shape, generator=generator),
        torch.randint(0, A, (size,), generator=generator),
        torch.rand(size, *shape, generator=generator),
        torch.ones(size),
    )


@pytest.mark.parametrize("actions", [True, False])
@pytest.mark.parametrize("observation", sorted(BASE))
def test_both_heads_act_on_predictions_of_the_temporal_embedding(observation, actions):
    q, ac = make(observation, "q_values", actions), make(observation, "actor_critic", actions)
    obs = torch.rand(5, *spec(observation, actions).shape)
    out = q(obs)
    assert isinstance(out, QValues) and out.q_values.shape == (5, A)
    z = q.encode(obs)
    assert z.shape == (5, 16)
    assert torch.allclose(out.q_values, q.net["value_decoder"](q.predictor(z)))
    policy = ac(obs)
    assert isinstance(policy, PolicyOutput) and policy.logits.shape == (5, A) and policy.value.shape == (5,)
    assert torch.allclose(policy.value, ac.net["value_decoder"](ac.encode(obs)))
    assert list(q.net) == [*trunk(actions), *TAIL, "value_decoder"]
    assert list(ac.net) == [*trunk(actions), *TAIL, "policy_decoder", "value_decoder"]


@pytest.mark.parametrize("observation", sorted(BASE))
def test_the_trunk_is_temporal_transformers_with_the_same_seeded_weights(observation):
    jepa = make(observation, "actor_critic")
    torch.manual_seed(0)
    plain = build_policy(dataclasses.replace(CONFIG, kind="temporal_transformer"), spec(observation), A, "actor_critic")
    shared = [name for name in plain.net if name in jepa.net and name not in ("policy_head", "value_head")]
    assert shared == [*trunk(), *TAIL[:-1]]
    for name in shared:
        assert all(torch.equal(a, b) for a, b in zip(plain.net[name].state_dict().values(), jepa.net[name].state_dict().values(), strict=True)), name
    obs = torch.rand(3, *spec(observation).shape)
    assert torch.equal(jepa.encode(obs), plain._pooled(obs, need_weights=False)[0])


@pytest.mark.parametrize("observation", sorted(BASE))
def test_the_embedding_depends_on_older_frames(observation):
    policy = make(observation).eval()
    obs = torch.rand(1, *spec(observation).shape)
    changed = obs.clone()
    changed[0, 0] += 1.0  # the oldest frame's first game channel
    with torch.no_grad():
        assert not torch.allclose(policy.encode(obs), policy.encode(changed))


@pytest.mark.parametrize("observation", sorted(BASE))
def test_attention_maps_cover_every_position_of_every_frame_and_the_action_tokens(observation):
    policy = make(observation)
    maps = policy.attention_maps(torch.rand(3, *spec(observation).shape))
    layout = policy.token_layout
    tokens = 1 + K * POSITIONS[observation] + K
    assert layout.num_frames == K and layout.tokens_per_frame == POSITIONS[observation] and layout.num_tokens == tokens
    assert KIND_INFO["jepa_temporal"].attention_view == "frames" and KIND_INFO["jepa_temporal"].uses_history
    assert len(maps) == CONFIG.num_layers  # num_layers: transformer blocks ...
    assert all(m.shape == (3, 2, tokens, tokens) for m in maps)
    assert len(policy.predictor.body) == len(make(observation, num_layers=1).predictor.body) + 1  # ... and predictor depth
    assert "TransformerBlock" in diagram(policy, "text") and "ActionPredictor" in diagram(policy, "text")


@pytest.mark.parametrize("actions", [True, False])
@pytest.mark.parametrize("observation", sorted(BASE))
def test_the_embedding_encoder_is_the_whole_observation_to_z_path_and_shares_the_weights(observation, actions):
    policy = make(observation, actions=actions)
    encoder = policy.embedding_encoder()
    obs = torch.rand(4, *spec(observation, actions).shape)
    assert torch.equal(encoder(obs), policy.encode(obs))
    policy_parameters = {id(p) for p in policy.parameters()}
    assert {id(p) for p in encoder.parameters()} <= policy_parameters
    encoder_names = [*trunk(actions), *TAIL[:-1]]
    size = sum(p.numel() for name in encoder_names for p in policy.net[name].parameters())
    assert sum(p.numel() for p in encoder.parameters()) == size
    assert not any(key.startswith(("front", "tail")) for key in policy.state_dict())  # never part of the policy's weights
    copied = copy.deepcopy(encoder)  # what the EMA target does: the encoder only, not the policy
    assert sum(p.numel() for p in copied.parameters()) == size
    assert not {id(p) for p in copied.parameters()} & policy_parameters
    assert torch.equal(copied(obs), policy.encode(obs))


def test_no_objective_without_the_coefficient():
    assert auxiliary_objective(make(prediction_coef=0.0)) is None
    assert auxiliary_objective(build_policy(ModelConfig(kind="jepa_temporal"), spec("vector"), A, "q_values")) is None


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
    assert set(terms.metrics) == {metric.key for metric in KIND_INFO["jepa_temporal"].metrics}


@pytest.mark.parametrize("trains_encoder", [True, False])
def test_the_prediction_loss_reaches_the_temporal_encoder_only_when_asked(trains_encoder):
    policy = make(prediction_trains_encoder=trains_encoder)
    auxiliary_objective(policy).loss(batch(policy)).loss.backward()
    encoder = [*trunk(), "cls", "blocks", "norm"]
    encoder_grad = sum(float(p.grad.abs().sum()) for name in encoder for p in policy.net[name].parameters() if p.grad is not None)
    assert (encoder_grad > 0) == trains_encoder
    assert policy.predictor.body[0].weight.grad.abs().sum() > 0


def tiny_dqn(model: dict, steps: int = 96, **dqn):
    return experiment_from_tables({
        "game": {"width": 4, "height": 4},
        "model": {"kind": "jepa_temporal", "d_model": 8, "num_heads": 2, "num_layers": 1, "prediction_coef": 1.0, **model},
        "train": {"algorithm": "dqn", "total_env_steps": steps, "device": "cpu", "log_interval": 32},
        "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8, **dqn},
        "env": {"num_envs": 2, "history_length": 2, "history_actions": True},
    })


@pytest.mark.parametrize("prioritized", [False, True])
@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_dqn_trains_the_objective(value_training, prioritized):
    config = tiny_dqn({"value_training": value_training}, prioritized=prioritized)
    env = make_vec_env(config)
    policy = make_policy(config, env)
    assert policy.history_length == 2 and policy.observation_spec.has_actions
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


def test_ppo_trains_the_prediction_loss_on_lunar_lander_with_hidden_velocities():
    pytest.importorskip("gymnasium")
    config = experiment_from_tables({
        "game": {"name": "lunar_lander", "hide_velocities": True},
        "model": {"kind": "jepa_temporal", "d_model": 8, "num_heads": 2, "num_layers": 1, "prediction_coef": 1.0},
        "train": {"total_env_steps": 64, "device": "cpu", "log_interval": 32},
        "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
        "env": {"num_envs": 4, "history_length": 3, "history_actions": True},
    })
    env = make_vec_env(config)
    trainer = make_trainer(config, make_policy(config, env), env, NullLogger())
    records = trainer.train()
    assert {"prediction_loss", "embedding_std"} <= set(records[-1]) and set(trainer.learner.extra_state()) == {STATE_KEY}
