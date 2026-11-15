"""``[model] residual``: hybrid, jepa_attention, frame_attention and lstm swap their conv / MLP stem for residual blocks, same module names."""

import dataclasses

import pytest
import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.config.fork_rules import locked_fields
from attentive_snake.models.config import OBJECTIVE_FIELDS, ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger
from attentive_snake.factory import make_trainer
from attentive_snake.viz import diagram

KINDS = ["hybrid", "jepa_attention", "frame_attention", "lstm"]
ATTENTIVE = ["hybrid", "jepa_attention", "frame_attention"]
STEM = {"hybrid": "stem", "jepa_attention": "stem", "frame_attention": "frame_encoder", "lstm": "frame_encoder"}  # the module that differs
OBSERVATIONS = ["grid", "vector"]
A = 4
SMALL = dict(d_model=16, num_heads=2, num_layers=1, conv_channels=8, conv_layers=2)


def spec(observation: str, kind: str) -> ObservationSpec:
    frames, actions = (3, A) if kind != "hybrid" else (1, 0)
    channels = (3 if observation == "grid" else 6) + actions
    shape = (frames * channels, 5, 5) if observation == "grid" else (frames * channels,)
    return ObservationSpec(observation, shape, frames, actions)


def make(kind: str, observation: str, residual: bool, head: str = "actor_critic"):
    torch.manual_seed(0)
    return build_policy(ModelConfig(kind=kind, residual=residual, **SMALL), spec(observation, kind), A, head)


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_forward_works_with_both_heads(kind, observation):
    obs = torch.rand(3, *spec(observation, kind).shape)
    assert isinstance(make(kind, observation, True)(obs), PolicyOutput)
    q = make(kind, observation, True, "q_values")(obs)
    assert isinstance(q, QValues) and q.q_values.shape == (3, A)


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_only_the_stem_differs_and_off_is_the_plain_network(kind, observation):
    plain, residual = make(kind, observation, False), make(kind, observation, True)
    plain_keys, residual_keys = set(plain.state_dict()), set(residual.state_dict())
    assert plain_keys != residual_keys
    assert {key for key in plain_keys ^ residual_keys if not key.startswith(f"net.{STEM[kind]}.")} == set()
    assert list(plain.net) == list(residual.net)  # same module names either way
    torch.manual_seed(0)
    default = build_policy(ModelConfig(kind=kind, **SMALL), spec(observation, kind), A)
    assert default.state_dict().keys() == plain.state_dict().keys()
    assert all(torch.equal(default.state_dict()[k], v) for k, v in plain.state_dict().items())  # seeded init unchanged too


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_only_the_kinds_with_a_stem_read_the_switch(kind, observation):
    reads = kind in KINDS
    assert ("residual" in KIND_INFO[kind].fields_for(observation)) == reads
    if not reads:
        build = lambda residual: build_policy(ModelConfig(kind=kind, residual=residual, **SMALL), spec(observation, "lstm"), A)
        assert build(False).state_dict().keys() == build(True).state_dict().keys()


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", ATTENTIVE)
def test_attention_maps_still_work(kind, observation):
    policy = make(kind, observation, True)
    layout = policy.token_layout
    maps = policy.attention_maps(torch.rand(2, *spec(observation, kind).shape))
    assert len(maps) == 1 and maps[0].shape == (2, 2, layout.num_tokens, layout.num_tokens)
    assert torch.allclose(maps[0].sum(-1), torch.ones(2, 2, layout.num_tokens), atol=1e-5)


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_the_diagram_draws_residual_networks(kind, observation):
    policy = make(kind, observation, True)
    for form in ("text", "mermaid", "html"):
        assert diagram(policy, form)
    assert "ResidualMlpBlock" in diagram(policy, "text") or "ResidualConvBlock" in diagram(policy, "text")


def test_residual_is_an_architecture_field_locked_on_fork():
    assert "residual" not in OBJECTIVE_FIELDS
    assert all("model.residual" in locked_fields(game) for game in ("snake", "lunar_lander"))


@pytest.mark.parametrize("game, history", [("snake", {"width": 4, "height": 4, "max_steps_without_food": 8}), ("lunar_lander", {})])
@pytest.mark.parametrize("kind", KINDS)
def test_a_few_ppo_updates_run(kind, game, history):
    config = experiment_from_tables({
        "game": {"name": game, **history},
        "model": {"kind": kind, "residual": True, **SMALL},
        "train": {"total_env_steps": 64, "device": "cpu", "log_interval": 32},
        "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
        "env": {"num_envs": 4, "history_length": 2, "history_actions": True},
    })
    env = make_vec_env(config)
    policy = make_policy(config, env)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    trainer = make_trainer(config, policy, env, NullLogger())
    assert trainer.train()
    assert any(not torch.equal(before[k], v) for k, v in policy.state_dict().items() if k.startswith(f"net.{STEM[kind]}."))
