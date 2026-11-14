import dataclasses

import pytest
import torch

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.policies.transformer import SnakeTransformerPolicy
from attentive_snake.models.types import PolicyOutput

NET_PARTS = ["embed", "positional", "cls", "blocks", "norm", "pool", "policy_head", "value_head"]


def make(shape=(3, 10, 10), num_actions=4, **kw) -> SnakeTransformerPolicy:
    return SnakeTransformerPolicy(ModelConfig(**kw), shape, num_actions)


@pytest.mark.parametrize("batch", [1, 2, 17])
def test_output_shapes_and_dtypes(batch):
    out = make()(torch.rand(batch, 3, 10, 10))
    assert isinstance(out, PolicyOutput)
    assert out.logits.shape == (batch, 4)
    assert out.value.shape == (batch,)
    assert out.logits.dtype == out.value.dtype == torch.float32


def test_non_square_grid_and_channels():
    m = make(shape=(5, 6, 9), num_actions=3)
    out = m(torch.rand(4, 5, 6, 9))
    assert out.logits.shape == (4, 3)
    assert out.value.shape == (4,)


def test_position_matters():
    m = make().eval()
    a = torch.zeros(1, 3, 10, 10)
    b = a.clone()
    a[0, 0, 0, 0] = 1
    b[0, 0, 5, 7] = 1
    with torch.no_grad():
        assert not torch.allclose(m(a).value, m(b).value)


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=65, num_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(num_layers=0)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ModelConfig().d_model = 8


def test_gradients_reach_all_parameters():
    m = make()
    out = m(torch.rand(8, 3, 10, 10))
    (out.logits.logsumexp(-1).sum() + out.value.sum()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_initial_policy_near_uniform():
    torch.manual_seed(0)
    m = make()
    with torch.no_grad():
        probs = m(torch.rand(32, 3, 10, 10)).logits.softmax(-1)
    assert torch.allclose(probs, torch.full_like(probs, 0.25), atol=0.02)


def test_net_has_named_parts():
    m = make(num_layers=2)
    for part in NET_PARTS:
        assert part in m.net
    assert len(m.net["blocks"]) == 2


def test_parameter_count_reasonable(capsys):
    n = sum(p.numel() for p in make().parameters())
    print(f"default parameter count: {n}")
    assert 1e5 < n < 1e6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    m = make().cuda()
    out = m(torch.rand(4, 3, 10, 10, device="cuda"))
    assert out.logits.is_cuda and out.value.shape == (4,)


def test_stores_env_description():
    m = make(shape=(5, 6, 9), num_actions=3)
    assert m.observation_shape == (5, 6, 9)
    assert m.num_actions == 3
    assert isinstance(m.config, ModelConfig)


def test_wrong_observation_shape_raises():
    with pytest.raises(ValueError, match="expected observations"):
        make()(torch.rand(2, 3, 9, 10))
    with pytest.raises(ValueError, match="expected observations"):
        make()(torch.rand(3, 10, 10))


def test_attention_maps_shapes_and_rows_sum_to_one():
    m = make(shape=(3, 5, 6), num_layers=2, num_heads=4)
    maps = m.attention_maps(torch.rand(3, 3, 5, 6))
    assert len(maps) == 2
    for a in maps:
        assert a.shape == (3, 4, 31, 31)
        assert torch.allclose(a.sum(-1), torch.ones(3, 4, 31), atol=1e-5)


def test_attention_maps_do_not_change_forward_or_mode():
    m = make(dropout=0.1)
    obs = torch.rand(2, 3, 10, 10)
    m.eval()
    before = m(obs)
    m.attention_maps(obs)
    assert not m.training
    after = m(obs)
    assert torch.equal(before.logits, after.logits)
    m.train()
    m.attention_maps(obs)
    assert m.training
    assert all(not a.requires_grad for a in m.attention_maps(obs))
