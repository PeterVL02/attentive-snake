"""Properties every selectable policy kind must satisfy."""

import pytest
import torch

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import POLICY_KINDS, build_policy
from attentive_snake.models.types import PolicyOutput
from attentive_snake.viz import diagram

H, W, A = 5, 6, 4
ATTENTIVE = ["transformer", "hybrid", "temporal_transformer", "frame_attention"]
TEMPORAL = ["temporal_transformer", "frame_attention"]
SMALL = dict(d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2)

# (history_length, history_actions)
VARIANTS = [(1, False), (4, False), (1, True), (4, True)]


def make(kind, k=1, actions=False, **kw):
    f = 3 + (A if actions else 0)
    return build_policy(ModelConfig(kind=kind, **{**SMALL, **kw}), (k * f, H, W), A, k)


def random_obs(k, actions, batch=3):
    """Random boards; with actions, frame 0 has no action, the others a random one."""
    f = 3 + (A if actions else 0)
    obs = torch.rand(batch, k * f, H, W)
    if actions:
        for b in range(batch):
            for j in range(k):
                base = j * f + 3
                obs[b, base : base + A] = 0
                if j > 0:
                    obs[b, base + (b + j) % A] = 1
    return obs


@pytest.mark.parametrize("kind", POLICY_KINDS)
@pytest.mark.parametrize("k, actions", VARIANTS)
def test_shapes_gradients_and_init(kind, k, actions):
    torch.manual_seed(0)
    m = make(kind, k, actions)
    obs = random_obs(k, actions, batch=32)
    out = m(obs)
    assert isinstance(out, PolicyOutput)
    assert out.logits.shape == (32, A) and out.value.shape == (32,)
    probs = out.logits.softmax(-1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / A), atol=0.03)
    (out.logits.logsumexp(-1).sum() + out.value.sum()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name  # no dead parameters


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_net_and_wrong_shape(kind):
    m = make(kind, 2, True)
    assert "policy_head" in m.net and "value_head" in m.net
    with pytest.raises(ValueError, match="expected observations"):
        m(torch.rand(2, 2 * 7, H + 1, W))
    with pytest.raises(ValueError, match="expected observations"):
        m(torch.rand(2 * 7, H, W))


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_diagram_renders(kind):
    for fmt in ("text", "mermaid", "html"):
        assert diagram(make(kind, 4, True), fmt)


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_indivisible_history_rejected(kind):
    with pytest.raises(ValueError, match="multiple of history_length"):
        build_policy(ModelConfig(kind=kind, **SMALL), (7, H, W), A, 2)


@pytest.mark.parametrize("kind", TEMPORAL)
def test_odd_frame_channels_rejected(kind):
    with pytest.raises(ValueError, match="channels"):
        build_policy(ModelConfig(kind=kind, **SMALL), (5, H, W), A, 1)


def test_cnn_has_no_attention():
    assert not hasattr(make("cnn"), "attention_maps")


@pytest.mark.parametrize("kind", ATTENTIVE)
@pytest.mark.parametrize("k, actions", VARIANTS)
def test_attention_maps_match_layout_and_sum_to_one(kind, k, actions):
    m = make(kind, k, actions)
    layout = m.token_layout
    maps = m.attention_maps(random_obs(k, actions))
    assert len(maps) == 2
    for a in maps:
        assert a.shape == (3, 2, layout.num_tokens, layout.num_tokens)
        assert torch.allclose(a.sum(-1), torch.ones(3, 2, layout.num_tokens), atol=1e-5)
        assert not a.requires_grad


@pytest.mark.parametrize("kind", ATTENTIVE)
def test_attention_maps_restore_mode(kind):
    m = make(kind, 2, True, dropout=0.1)
    obs = random_obs(2, True)
    m.eval()
    before = m(obs)
    m.attention_maps(obs)
    assert not m.training and torch.equal(before.logits, m(obs).logits)
    m.train()
    m.attention_maps(obs)
    assert m.training


def test_temporal_token_layout():
    layout = make("temporal_transformer", 4, True).token_layout
    assert (layout.num_frames, layout.tokens_per_frame, layout.num_action_tokens) == (4, H * W, 4)
    assert layout.num_tokens == 1 + 4 * H * W + 4
    assert layout.frame_slice(3) == slice(1 + 3 * H * W, 1 + 4 * H * W)
    assert layout.action_index(0) == 1 + 4 * H * W
    no_actions = make("temporal_transformer", 4, False).token_layout
    assert no_actions.num_tokens == 1 + 4 * H * W
    with pytest.raises(ValueError):
        no_actions.action_index(0)


def test_frame_attention_token_layout():
    layout = make("frame_attention", 4, True).token_layout
    assert (layout.num_frames, layout.tokens_per_frame, layout.num_tokens) == (4, 1, 5)


@pytest.mark.parametrize("kind", TEMPORAL)
def test_action_planes_change_output(kind):
    torch.manual_seed(0)
    m = make(kind, 3, True).eval()
    obs = random_obs(3, True, batch=1)
    changed = obs.clone()
    f = 7
    newest = 2 * f + 3
    old = int(obs[0, newest : newest + A].argmax())
    changed[0, newest : newest + A] = 0
    changed[0, newest + (old + 1) % A] = 1
    with torch.no_grad():
        assert not torch.allclose(m(obs).value, m(changed).value)
        # "no action" is distinct from every real action
        changed[0, newest : newest + A] = 0
        assert not torch.allclose(m(obs).value, m(changed).value)


@pytest.mark.parametrize("kind", TEMPORAL)
def test_action_read_from_any_cell_and_ignores_board_for_action(kind):
    from attentive_snake.models.frames import action_ids, split_frames

    m = make(kind, 2, True)
    obs = random_obs(2, True, batch=2)
    ids = action_ids(split_frames(obs, m.frame_spec), A)
    assert ids[:, 0].tolist() == [A, A] and (ids[:, 1] < A).all()


@pytest.mark.parametrize("kind", TEMPORAL)
def test_temporal_order_matters(kind):
    torch.manual_seed(0)
    m = make(kind, 2, False).eval()
    obs = random_obs(2, False, batch=1)
    swapped = torch.cat([obs[:, 3:], obs[:, :3]], dim=1)
    with torch.no_grad():
        assert not torch.allclose(m(obs).value, m(swapped).value)


def test_registry_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        ModelConfig(kind="mlp")
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="unknown model kind"):
        build_policy(SimpleNamespace(kind="mlp"), (3, H, W), A)
    assert set(POLICY_KINDS) == {"transformer", "cnn", "hybrid", "temporal_transformer", "frame_attention"}


def test_cnn_config_validation():
    with pytest.raises(ValueError, match="odd"):
        ModelConfig(kernel_size=4)
    with pytest.raises(ValueError):
        ModelConfig(conv_layers=0)
