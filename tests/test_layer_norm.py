"""``[model] layer_norm``: normalisation in the conv / MLP trunks, stems, frame encoders and readouts, under the same module names."""

import pytest
import torch
from torch import nn

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.fork_rules import locked_fields
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.blocks.norm import map_norm, normalised, present, vector_norm
from attentive_snake.models.config import OBJECTIVE_FIELDS, ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger
from attentive_snake.viz import diagram

KINDS = ["cnn", "resnet", "mlp", "jepa", "hybrid", "jepa_attention", "jepa_tokens", "frame_attention", "lstm"]
UNCHANGED = ["transformer", "temporal_transformer", "jepa_temporal"]  # nothing but transformer blocks, which have layer norms already
TRUNK = {  # the children of ``net`` that gain normalisation
    "cnn": ("encoder", "readout"),
    "resnet": ("encoder", "readout"),
    "mlp": ("encoder",),
    "jepa": ("encoder", "readout"),
    "hybrid": ("stem",),
    "jepa_attention": ("stem",),
    "jepa_tokens": ("stem",),
    "frame_attention": ("frame_encoder",),
    "lstm": ("frame_encoder",),
}
OBSERVATIONS = ["grid", "vector"]
NORMS = (nn.LayerNorm, nn.GroupNorm)
A = 4
SMALL = dict(d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2)


def spec(observation: str, kind: str) -> ObservationSpec:
    frames, actions = (3, A) if KIND_INFO[kind].uses_history else (1, 0)
    channels = (3 if observation == "grid" else 6) + actions
    shape = (frames * channels, 5, 5) if observation == "grid" else (frames * channels,)
    return ObservationSpec(observation, shape, frames, actions)


def make(kind: str, observation: str, layer_norm: bool, head: str = "actor_critic", **model):
    torch.manual_seed(0)
    return build_policy(ModelConfig(kind=kind, layer_norm=layer_norm, **(SMALL | model)), spec(observation, kind), A, head)


def norms(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, NORMS)]


def test_the_two_norms_and_their_absence():
    assert vector_norm(8, False) is None and map_norm(8, False) is None and present(None, None) == []
    layer, group = vector_norm(8, True), map_norm(8, True)
    assert isinstance(layer, nn.LayerNorm) and isinstance(group, nn.GroupNorm) and group.num_groups == 1
    assert present(None, layer) == [layer]
    x = torch.rand(2, 8, 3, 3)
    assert normalised(None, x) is x
    out = normalised(group, x * 50 + 7)  # each board over all its channels and cells: mean 0, variance 1
    assert torch.allclose(out.mean(dim=(1, 2, 3)), torch.zeros(2), atol=1e-5) and torch.allclose(out.var(dim=(1, 2, 3), correction=0), torch.ones(2), atol=1e-3)


@pytest.mark.parametrize("head", ["actor_critic", "q_values", "dueling"])
@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_forward_works_with_every_head(kind, observation, head):
    obs = torch.rand(3, *spec(observation, kind).shape)
    out = make(kind, observation, True, head)(obs)
    values = out.logits if isinstance(out, PolicyOutput) else out.q_values
    assert isinstance(out, PolicyOutput if head == "actor_critic" else QValues)
    assert values.shape == (3, A) and torch.isfinite(values).all()


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_off_is_the_network_from_before_the_switch_and_on_adds_norms_in_the_trunk_only(kind, observation):
    off, on = make(kind, observation, False), make(kind, observation, True)
    torch.manual_seed(0)
    default = build_policy(ModelConfig(kind=kind, **SMALL), spec(observation, kind), A)
    assert default.state_dict().keys() == off.state_dict().keys()
    assert all(torch.equal(default.state_dict()[k], v) for k, v in off.state_dict().items())  # seeded init unchanged too
    assert diagram(default, "text") == diagram(off, "text")  # no placeholder modules either
    assert list(off.net) == list(on.net)  # same module names either way
    added = set(on.state_dict()) - set(off.state_dict())
    assert set(off.state_dict()) < set(on.state_dict())  # every old key is still there
    assert added and all(key.startswith(tuple(f"net.{name}." for name in TRUNK[kind])) for key in added)
    assert all(on.state_dict()[key].ndim == 1 for key in added)  # scales and shifts only
    assert len(norms(on)) > len(norms(off))
    already_normed = (kind, observation) == ("resnet", "vector")  # its residual MLP blocks have layer norms either way
    for name in TRUNK[kind]:
        assert len(norms(on.net[name])) > len(norms(off.net[name])) or (already_normed and name == "encoder"), name
    assert not [m for m in on.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@pytest.mark.parametrize("kind", ["cnn", "resnet", "jepa", "hybrid", "frame_attention", "lstm"])
def test_grid_feature_maps_get_a_one_group_group_norm_after_every_convolution(kind):
    policy = make(kind, "grid", True, residual=kind in ("hybrid", "lstm"))
    trunk = nn.ModuleList(policy.net[name] for name in TRUNK[kind])
    groups = [m for m in trunk.modules() if isinstance(m, nn.GroupNorm)]
    convs = [m for m in trunk.modules() if isinstance(m, nn.Conv2d)]
    assert groups and len(groups) == len(convs) and all(m.num_groups == 1 and m.num_channels == 8 for m in groups)
    if "readout" in dict(policy.net.named_children()) or kind in ("frame_attention", "lstm"):
        assert [m for m in trunk.modules() if isinstance(m, nn.LayerNorm)]  # the readout's


@pytest.mark.parametrize("kind", KINDS)
def test_vector_layers_get_a_layer_norm_each(kind):
    policy = make(kind, "vector", True)
    trunk = nn.ModuleList(policy.net[name] for name in TRUNK[kind])
    assert not [m for m in trunk.modules() if isinstance(m, nn.GroupNorm)]
    layers = 2  # SMALL: conv_layers = num_layers = 2 (resnet: its residual blocks' own norms plus a final one)
    expected = {"mlp": layers, "hybrid": layers, "jepa_attention": layers, "jepa_tokens": layers, "resnet": layers + 1 + 1}
    assert len(norms(trunk)) == expected.get(kind, layers + 1)  # + the readout


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_exactly_the_kinds_with_a_conv_or_mlp_part_read_the_switch(kind, observation):
    reads = kind in KINDS
    assert set(KINDS) | set(UNCHANGED) == set(POLICY_KINDS)
    assert ("layer_norm" in KIND_INFO[kind].fields_for(observation)) == reads
    if not reads:
        assert make(kind, observation, False).state_dict().keys() == make(kind, observation, True).state_dict().keys()


def test_residual_vector_blocks_keep_their_own_norms_so_only_a_readout_changes():
    # the documented exceptions of the help text
    assert make("hybrid", "vector", False, residual=True).state_dict().keys() == make("hybrid", "vector", True, residual=True).state_dict().keys()
    for kind in ("lstm", "frame_attention"):
        off, on = make(kind, "vector", False, residual=True), make(kind, "vector", True, residual=True)
        assert set(on.state_dict()) - set(off.state_dict()) == {"net.frame_encoder.readout.norm.weight", "net.frame_encoder.readout.norm.bias"}
    off, on = make("resnet", "vector", False), make("resnet", "vector", True)
    assert set(on.state_dict()) - set(off.state_dict()) == {"net.readout.norm.weight", "net.readout.norm.bias"}


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_an_observations_output_does_not_depend_on_the_rest_of_the_batch(kind, observation):
    policy = make(kind, observation, True).train()
    obs = torch.rand(4, *spec(observation, kind).shape)
    torch.testing.assert_close(policy(obs).logits[:1], policy(obs[:1]).logits, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", ["hybrid", "jepa_attention", "jepa_tokens", "frame_attention"])
def test_attention_maps_still_work(kind, observation):
    policy = make(kind, observation, True)
    layout = policy.token_layout
    maps = policy.attention_maps(torch.rand(2, *spec(observation, kind).shape))
    assert len(maps) == 2 and maps[0].shape == (2, 2, layout.num_tokens, layout.num_tokens)


@pytest.mark.parametrize("observation", OBSERVATIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_the_diagram_draws_the_norms(kind, observation):
    policy = make(kind, observation, True)
    for form in ("text", "mermaid", "html"):
        assert diagram(policy, form)
    assert "GroupNorm" in diagram(policy, "text") or "LayerNorm" in diagram(policy, "text")


def test_layer_norm_is_an_architecture_field_locked_on_fork():
    assert "layer_norm" not in OBJECTIVE_FIELDS
    assert all("model.layer_norm" in locked_fields(game) for game in ("snake", "lunar_lander"))
    assert all("train.weight_decay" not in locked_fields(game) for game in ("snake", "lunar_lander"))


ALGORITHMS = {
    "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
    "dqn": {"buffer_size": 64, "batch_size": 4, "learning_starts": 8, "target_update_interval": 16},
}


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("game, rules", [("snake", {"width": 4, "height": 4, "max_steps_without_food": 8}), ("lunar_lander", {})])
@pytest.mark.parametrize("kind", KINDS)
def test_a_few_updates_train_the_norms(kind, game, rules, algorithm):
    config = experiment_from_tables({
        "game": {"name": game, **rules},
        "model": {"kind": kind, "layer_norm": True, **SMALL},
        "train": {"algorithm": algorithm, "total_env_steps": 64, "device": "cpu", "log_interval": 32},
        algorithm: ALGORITHMS[algorithm],
        "env": {"num_envs": 4, "history_length": 2, "history_actions": True},
    })
    env = make_vec_env(config)
    policy = make_policy(config, env)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    trainer = make_trainer(config, policy, env, NullLogger())
    assert trainer.train()
    norm_keys = [f"{name}.weight" for name, module in policy.named_modules() if isinstance(module, NORMS) and name.startswith(tuple(f"net.{t}." for t in TRUNK[kind]))]
    assert norm_keys and any(not torch.equal(before[k], policy.state_dict()[k]) for k in norm_keys)
