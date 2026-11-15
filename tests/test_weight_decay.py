"""``[train] weight_decay``: which parameters decay, Adam at 0 and AdamW above, and restoring the optimizer on resume / fork."""

import copy

import pytest
import torch
from torch import nn

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.training import NullLogger, TrainConfig, fork_from_checkpoint, latest_checkpoint, resume_from_checkpoint
from attentive_snake.training.optimizer import build_optimizer, decays, parameter_order
from attentive_snake.training.optimizer_state import restore_optimizer
from attentive_snake.training.runtime import seed_everything
from attentive_snake.training.weight_decay import decay_groups, decayed_names

A = 4
SMALL = dict(d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2)
ALGORITHMS = {
    "ppo": {"rollout_length": 8, "num_epochs": 2, "num_minibatches": 2},
    "dqn": {"buffer_size": 120, "batch_size": 8, "learning_starts": 16, "target_update_interval": 40},
}
LEARNER_MODULES = {"ppo": "attentive_snake.training.ppo.learner", "dqn": "attentive_snake.training.dqn.learner"}
OLD_EPS = {"ppo": 1e-5, "dqn": 1e-8}  # what the learners passed to Adam before the setting existed


def policy_of(kind: str, observation: str, head: str = "actor_critic", **model) -> nn.Module:
    temporal = KIND_INFO[kind].uses_history
    frames, actions = (3, A) if temporal else (1, 0)
    channels = (3 if observation == "grid" else 6) + actions
    shape = (frames * channels, 5, 5) if observation == "grid" else (frames * channels,)
    return build_policy(ModelConfig(kind=kind, **(SMALL | model)), ObservationSpec(observation, shape, frames, actions), A, head)


def tables(algorithm: str, **train) -> dict:
    return {
        "game": {"name": "snake", "width": 5, "height": 5, "max_steps_without_food": 20},
        "model": {"kind": "cnn", "d_model": 16, "conv_channels": 8, "conv_layers": 2, "layer_norm": True},
        "train": {"algorithm": algorithm, "total_env_steps": 160, "device": "cpu", "seed": 3, "log_interval": 1, **train},
        algorithm: ALGORITHMS[algorithm],
        "env": {"num_envs": 4},
    }


def trainer_of(algorithm: str, **train):
    config = experiment_from_tables(tables(algorithm, **train))
    env = make_vec_env(config)
    seed_everything(config.train.seed)
    return make_trainer(config, make_policy(config, env), env, NullLogger())


def trained_weights(algorithm: str, **train) -> dict[str, torch.Tensor]:
    trainer = trainer_of(algorithm, **train)
    trainer.train()
    return {name: value.clone() for name, value in trainer.policy.state_dict().items()}


# --- which parameters decay -----------------------------------------------------------------


@pytest.mark.parametrize("head", ["actor_critic", "q_values", "dueling"])
@pytest.mark.parametrize("observation", ["grid", "vector"])
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_every_parameter_is_in_exactly_one_group_and_only_weight_matrices_decay(kind, observation, head):
    policy = policy_of(kind, observation, head, layer_norm=True, residual=True)
    groups = decay_groups(policy)
    everything = [id(p) for p in policy.parameters()]
    assert sorted(id(p) for p in groups.ordered()) == sorted(everything) and len(set(everything)) == len(everything)
    names = decayed_names(policy)
    assert names and names <= dict(policy.named_parameters()).keys()
    for name, parameter in policy.named_parameters():
        owner = policy.get_submodule(name.rpartition(".")[0])
        if parameter.ndim < 2 or isinstance(owner, (nn.Embedding, nn.LayerNorm, nn.GroupNorm)):
            assert name not in names, name  # biases, normalisation, embedding tables
        if isinstance(owner, (nn.Linear, nn.Conv2d)) and name.endswith(".weight"):
            assert name in names, name
    assert not [name for name in names if "bias" in name]


def test_tokens_positions_embeddings_and_norms_are_exempt_and_attention_and_lstm_weights_decay():
    transformer = decayed_names(policy_of("temporal_transformer", "grid"))
    exempt = dict(policy_of("temporal_transformer", "grid").named_parameters()).keys() - transformer
    assert {"net.cls.token", "net.positional.row", "net.positional.col", "net.temporal.frame", "net.action_embed.table.weight"} <= exempt
    assert {"net.norm.weight", "net.blocks.0.norm1.weight", "net.blocks.0.attention.attn.in_proj_bias"} <= exempt
    assert {"net.embed.proj.weight", "net.blocks.0.attention.attn.in_proj_weight", "net.blocks.0.attention.attn.out_proj.weight", "net.blocks.0.feed_forward.fc1.weight", "net.policy_head.linear.weight"} <= transformer

    vector = policy_of("hybrid", "vector")
    assert "net.stem.positional.element" not in decayed_names(vector) and "net.stem.embed.proj.weight" in decayed_names(vector)

    lstm = decayed_names(policy_of("lstm", "grid"))
    assert {"net.recurrent.lstm.weight_ih_l0", "net.recurrent.lstm.weight_hh_l0", "net.recurrent.lstm.weight_ih_l1"} <= lstm
    assert not {name for name in lstm if "bias" in name}

    jepa = policy_of("jepa", "grid", "q_values", layer_norm=True)
    assert "net.predictor.action_embedding.weight" not in decayed_names(jepa)
    assert {"net.encoder.stem.1.weight", "net.readout.norm.weight"} <= dict(jepa.named_parameters()).keys() - decayed_names(jepa)  # the norms' scales
    assert "net.encoder.stem.0.weight" in decayed_names(jepa)


def test_a_parameter_of_an_unknown_kind_of_module_is_not_decayed():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.ones(3, 3))
            self.linear = nn.Linear(3, 3)

    assert decayed_names(Odd()) == {"linear.weight"}


# --- the optimizer --------------------------------------------------------------------------


def test_zero_weight_decay_is_plain_adam_over_the_parameters_in_their_own_order():
    policy = policy_of("hybrid", "grid")
    optimizer = build_optimizer(policy, TrainConfig(learning_rate=1e-3), eps=1e-5)
    assert type(optimizer) is torch.optim.Adam and not decays(optimizer.param_groups)
    (group,) = optimizer.param_groups
    assert [id(p) for p in group["params"]] == [id(p) for p in policy.parameters()]
    assert group["lr"] == 1e-3 and group["eps"] == 1e-5 and group["weight_decay"] == 0


def test_weight_decay_above_zero_is_adamw_with_a_decayed_and_an_exempt_group():
    policy = policy_of("hybrid", "grid")
    optimizer = build_optimizer(policy, TrainConfig(learning_rate=1e-3, weight_decay=0.01), eps=1e-8)
    assert type(optimizer) is torch.optim.AdamW and decays(optimizer.param_groups)
    decayed, exempt = optimizer.param_groups
    assert (decayed["weight_decay"], exempt["weight_decay"]) == (0.01, 0.0)
    assert decayed["lr"] == exempt["lr"] == 1e-3 and decayed["eps"] == exempt["eps"] == 1e-8
    groups = decay_groups(policy)
    assert [id(p) for p in decayed["params"]] == [id(p) for p in groups.decayed]
    assert [id(p) for p in exempt["params"]] == [id(p) for p in groups.exempt]
    assert [id(p) for p in parameter_order(policy, True)] == [id(p) for p in groups.ordered()]


def test_a_step_without_gradient_shrinks_exactly_the_decayed_weights():
    policy = policy_of("temporal_transformer", "vector")
    before = {name: value.clone() for name, value in policy.named_parameters()}
    optimizer = build_optimizer(policy, TrainConfig(learning_rate=0.1, weight_decay=0.5), eps=1e-8)
    sum((p * 0.0).sum() for p in policy.parameters()).backward()
    optimizer.step()
    names = decayed_names(policy)
    for name, value in policy.named_parameters():
        expected = before[name] * (1 - 0.1 * 0.5) if name in names else before[name]
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=0)


def test_negative_weight_decay_is_refused():
    with pytest.raises(ValueError, match="weight_decay"):
        TrainConfig(weight_decay=-0.1)


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_zero_weight_decay_trains_exactly_as_the_adam_the_learner_built_before(algorithm, monkeypatch):
    weights = trained_weights(algorithm)
    assert weights.keys() == trained_weights(algorithm, weight_decay=0.0).keys()

    def old_optimizer(policy, train, eps):
        assert eps == OLD_EPS[algorithm]
        return torch.optim.Adam(policy.parameters(), lr=train.learning_rate, eps=eps)

    module = __import__(LEARNER_MODULES[algorithm], fromlist=["build_optimizer"])
    monkeypatch.setattr(module, "build_optimizer", old_optimizer)
    before = trained_weights(algorithm)
    assert all(torch.equal(weights[name], value) for name, value in before.items())


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_weight_decay_changes_training_and_uses_adamw(algorithm):
    trainer = trainer_of(algorithm, weight_decay=0.1)
    assert type(trainer.learner.optimizer) is torch.optim.AdamW
    trainer.train()
    plain = trained_weights(algorithm)
    decayed = trainer.policy.state_dict()
    assert any(not torch.equal(plain[name], value) for name, value in decayed.items())
    assert all(torch.isfinite(value).all() for value in decayed.values())


# --- resume and fork ------------------------------------------------------------------------


def moments(trainer) -> dict[str, dict[str, torch.Tensor]]:
    """Each parameter's optimizer state by parameter name."""
    return {name: trainer.learner.optimizer.state[parameter] for name, parameter in trainer.policy.named_parameters()}


def assert_same_moments(a, b) -> None:
    first, second = moments(a), moments(b)
    assert first.keys() == second.keys() and first
    for name in first:
        assert first[name].keys() == second[name].keys() == {"step", "exp_avg", "exp_avg_sq"}
        for key in first[name]:
            assert torch.equal(torch.as_tensor(first[name][key]), torch.as_tensor(second[name][key])), (name, key)


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("parent_decay, child_decay", [(0.0, 0.01), (0.01, 0.0), (0.01, 0.05), (0.0, 0.0)])
def test_a_fork_keeps_every_parameters_moments_and_takes_its_own_weight_decay(tmp_path, algorithm, parent_decay, child_decay):
    parent = trainer_of(algorithm, weight_decay=parent_decay, checkpoint_dir=tmp_path / "parent")
    parent.train()
    child = trainer_of(algorithm, weight_decay=child_decay, learning_rate=1e-3, checkpoint_dir=tmp_path / "child")
    assert fork_from_checkpoint(child, latest_checkpoint(tmp_path / "parent")) == 160
    assert_same_moments(parent, child)
    optimizer = child.learner.optimizer
    assert type(optimizer) is (torch.optim.AdamW if child_decay else torch.optim.Adam)
    assert [group["weight_decay"] for group in optimizer.param_groups] == ([child_decay, 0.0] if child_decay else [0])
    assert all(group["lr"] == 1e-3 for group in optimizer.param_groups)  # the child's settings, not the checkpoint's
    assert child.train() and child.env_steps == 160


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_resume_with_weight_decay_restores_the_optimizer(tmp_path, algorithm):
    first = trainer_of(algorithm, weight_decay=0.01, checkpoint_dir=tmp_path)
    first.train()
    again = trainer_of(algorithm, weight_decay=0.01, total_env_steps=320, checkpoint_dir=tmp_path)
    assert resume_from_checkpoint(again, latest_checkpoint(tmp_path)) == 160
    assert_same_moments(first, again)
    assert [group["weight_decay"] for group in again.learner.optimizer.param_groups] == [0.01, 0.0]
    assert again.train() and again.env_steps == 320


def test_restoring_into_the_same_layout_equals_loading_the_state_dict():
    policy = policy_of("cnn", "grid")
    train = TrainConfig(learning_rate=1e-3)
    trained = build_optimizer(policy, train, eps=1e-5)
    policy(torch.rand(2, 3, 5, 5)).logits.sum().backward()
    trained.step()
    saved = copy.deepcopy(trained.state_dict())
    restored, loaded = build_optimizer(policy, train, eps=1e-5), build_optimizer(policy, train, eps=1e-5)
    restore_optimizer(restored, saved, policy)
    loaded.load_state_dict(saved)
    a, b = restored.state_dict(), loaded.state_dict()
    assert a["param_groups"] == b["param_groups"] and a["state"].keys() == b["state"].keys()
    assert all(torch.equal(a["state"][i]["exp_avg_sq"], b["state"][i]["exp_avg_sq"]) for i in a["state"])


def test_an_optimizer_state_of_another_network_is_refused():
    small, large = policy_of("cnn", "grid"), policy_of("hybrid", "grid")
    saved = build_optimizer(large, TrainConfig(), eps=1e-5).state_dict()
    with pytest.raises(ValueError, match="does not fit"):
        restore_optimizer(build_optimizer(small, TrainConfig(), eps=1e-5), saved, small)
