"""Properties every selectable policy kind must satisfy, for grid and vector observations."""

import json
from pathlib import Path

import pytest
import torch

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.models.types import PolicyOutput
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.viz import diagram

H, W, A = 5, 6, 4
BASE = {"grid": 3, "vector": 5}  # game channels per frame: board planes / numbers
ATTENTIVE = ["transformer", "hybrid", "temporal_transformer", "frame_attention"]
TEMPORAL = ["temporal_transformer", "frame_attention", "lstm"]
NO_ATTENTION = ["cnn", "resnet", "lstm", "mlp"]
OBSERVATIONS = ["grid", "vector"]
SMALL = dict(d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2)

# (history_length, history_actions)
VARIANTS = [(1, False), (4, False), (1, True), (4, True)]


def frame_size(obs_kind, actions):
    return BASE[obs_kind] + (A if actions else 0)


def spec(obs_kind="grid", k=1, actions=False):
    channels = k * frame_size(obs_kind, actions)
    shape = (channels, H, W) if obs_kind == "grid" else (channels,)
    return ObservationSpec(obs_kind, shape, k, A if actions else 0)


def make(kind, obs_kind="grid", k=1, actions=False, **kw):
    return build_policy(ModelConfig(kind=kind, **{**SMALL, **kw}), spec(obs_kind, k, actions), A)


def random_obs(obs_kind, k, actions, batch=3):
    """Random observations; with actions, frame 0 has no action, the others a random one."""
    f = frame_size(obs_kind, actions)
    obs = torch.rand(batch, *spec(obs_kind, k, actions).shape)
    if actions:
        for b in range(batch):
            for j in range(k):
                first = j * f + BASE[obs_kind]
                obs[b, first : first + A] = 0
                if j > 0:
                    obs[b, first + (b + j) % A] = 1
    return obs


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", POLICY_KINDS)
@pytest.mark.parametrize("k, actions", VARIANTS)
def test_shapes_gradients_and_init(kind, obs_kind, k, actions):
    torch.manual_seed(0)
    m = make(kind, obs_kind, k, actions)
    obs = random_obs(obs_kind, k, actions, batch=32)
    out = m(obs)
    assert isinstance(out, PolicyOutput)
    assert out.logits.shape == (32, A) and out.value.shape == (32,)
    probs = out.logits.softmax(-1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / A), atol=0.03)
    (out.logits.logsumexp(-1).sum() + out.value.sum()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if kind == "lstm" and k == 1 and "weight_hh" in name:
            continue  # a one-frame window never feeds the recurrent weights: the hidden state before it is zero
        assert p.grad.abs().sum() > 0, name  # no dead parameters


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_net_and_wrong_shape(kind, obs_kind):
    m = make(kind, obs_kind, 2, True)
    assert "policy_head" in m.net and "value_head" in m.net
    good = spec(obs_kind, 2, True).shape
    with pytest.raises(ValueError, match="expected observations"):
        m(torch.rand(2, good[0] + 1, *good[1:]))
    with pytest.raises(ValueError, match="expected observations"):
        m(torch.rand(*good))


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_diagram_renders(kind, obs_kind):
    for fmt in ("text", "mermaid", "html"):
        assert diagram(make(kind, obs_kind, 4, True), fmt)


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_indivisible_history_rejected(kind):
    with pytest.raises(ValueError, match="multiple of history_length"):
        build_policy(ModelConfig(kind=kind, **SMALL), ObservationSpec("grid", (7, H, W), 2), A)


@pytest.mark.parametrize("kind", TEMPORAL)
def test_action_channels_must_be_one_per_action(kind):
    with pytest.raises(ValueError, match="action channels"):
        build_policy(ModelConfig(kind=kind, **SMALL), ObservationSpec("grid", (5, H, W), 1, action_channels=2), A)


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_unsupported_observation_kind_rejected(kind):
    with pytest.raises(ValueError, match="cannot read image observations"):
        build_policy(ModelConfig(kind=kind, **SMALL), ObservationSpec("image", (3, 8, 8)), A)


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_kind_info_observation_kinds_match_the_policy(kind):
    m = make(kind)
    assert KIND_INFO[kind].observation_kinds == type(m).observation_kinds == ("grid", "vector")


def test_base_channels_come_from_the_spec_not_the_channel_count():
    """Five game channels plus four action planes per frame: nothing assumes Snake's three board channels."""
    five = ObservationSpec("grid", (2 * 9, H, W), 2, action_channels=A)
    for kind in TEMPORAL:
        m = build_policy(ModelConfig(kind=kind, **SMALL), five, A)
        assert m(torch.rand(2, 18, H, W)).logits.shape == (2, A)


@pytest.mark.parametrize("kind", NO_ATTENTION)
def test_kinds_without_attention_have_no_attention_maps(kind):
    assert not hasattr(make(kind), "attention_maps")


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", ATTENTIVE)
@pytest.mark.parametrize("k, actions", VARIANTS)
def test_attention_maps_match_layout_and_sum_to_one(kind, obs_kind, k, actions):
    m = make(kind, obs_kind, k, actions)
    layout = m.token_layout
    maps = m.attention_maps(random_obs(obs_kind, k, actions))
    assert len(maps) == 2
    for a in maps:
        assert a.shape == (3, 2, layout.num_tokens, layout.num_tokens)
        assert torch.allclose(a.sum(-1), torch.ones(3, 2, layout.num_tokens), atol=1e-5)
        assert not a.requires_grad


@pytest.mark.parametrize("kind", ATTENTIVE)
def test_attention_maps_restore_mode(kind):
    m = make(kind, "grid", 2, True, dropout=0.1)
    obs = random_obs("grid", 2, True)
    m.eval()
    before = m(obs)
    m.attention_maps(obs)
    assert not m.training and torch.equal(before.logits, m(obs).logits)
    m.train()
    m.attention_maps(obs)
    assert m.training


def test_temporal_token_layout():
    layout = make("temporal_transformer", "grid", 4, True).token_layout
    assert (layout.num_frames, layout.tokens_per_frame, layout.num_action_tokens) == (4, H * W, 4)
    assert layout.num_tokens == 1 + 4 * H * W + 4
    assert layout.frame_slice(3) == slice(1 + 3 * H * W, 1 + 4 * H * W)
    assert layout.action_index(0) == 1 + 4 * H * W
    no_actions = make("temporal_transformer", "grid", 4, False).token_layout
    assert no_actions.num_tokens == 1 + 4 * H * W
    with pytest.raises(ValueError):
        no_actions.action_index(0)
    vector = make("temporal_transformer", "vector", 4, True).token_layout
    assert (vector.num_frames, vector.tokens_per_frame, vector.num_action_tokens) == (4, BASE["vector"], 4)


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
def test_frame_attention_token_layout(obs_kind):
    layout = make("frame_attention", obs_kind, 4, True).token_layout
    assert (layout.num_frames, layout.tokens_per_frame, layout.num_tokens) == (4, 1, 5)


@pytest.mark.parametrize("kind, k, actions", [("transformer", 1, True), ("transformer", 3, False), ("hybrid", 2, True)])
def test_vector_element_tokens_are_truthful(kind, k, actions):
    """Changing element e of any frame changes exactly token e (frames are extra values of each element's token)."""
    torch.manual_seed(0)
    m = make(kind, "vector", k, actions).eval()
    layout = m.token_layout
    f = frame_size("vector", actions)
    assert (layout.num_frames, layout.tokens_per_frame) == (1, f)
    obs = random_obs("vector", k, actions, batch=1)
    for frame in range(k):
        for element in range(f):
            changed = obs.clone()
            changed[0, frame * f + element] += 1.0
            diff = (m._tokens(changed) - m._tokens(obs)).abs().sum(-1)[0]
            assert diff.shape == (layout.num_tokens - 1,)
            assert diff.nonzero().flatten().tolist() == [layout.frame_slice(0).start - 1 + element]


def test_temporal_vector_tokens_are_truthful():
    """Changing game element e of frame j changes exactly the token at frame_slice(j)[e]."""
    torch.manual_seed(0)
    k = 3
    m = make("temporal_transformer", "vector", k, True).eval()
    layout = m.token_layout
    f = frame_size("vector", True)
    obs = random_obs("vector", k, True, batch=1)
    for frame in range(k):
        for element in range(BASE["vector"]):
            changed = obs.clone()
            changed[0, frame * f + element] += 1.0
            diff = (m._tokens(changed) - m._tokens(obs)).abs().sum(-1)[0]
            assert diff.nonzero().flatten().tolist() == [layout.frame_slice(frame).start - 1 + element]


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", TEMPORAL)
def test_action_planes_change_output(kind, obs_kind):
    torch.manual_seed(0)
    m = make(kind, obs_kind, 3, True).eval()
    obs = random_obs(obs_kind, 3, True, batch=1)
    changed = obs.clone()
    f = frame_size(obs_kind, True)
    newest = 2 * f + BASE[obs_kind]
    old = int(obs[0, newest : newest + A].reshape(A, -1)[:, 0].argmax())
    changed[0, newest : newest + A] = 0
    changed[0, newest + (old + 1) % A] = 1
    with torch.no_grad():
        assert not torch.allclose(m(obs).value, m(changed).value)
        # "no action" is distinct from every real action
        changed[0, newest : newest + A] = 0
        assert not torch.allclose(m(obs).value, m(changed).value)


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", TEMPORAL)
def test_action_read_from_any_cell_and_ignores_board_for_action(kind, obs_kind):
    from attentive_snake.models.frames import action_ids, split_frames

    m = make(kind, obs_kind, 2, True)
    obs = random_obs(obs_kind, 2, True, batch=2)
    ids = action_ids(split_frames(obs, m.observation_spec), m.observation_spec, A)
    assert ids[:, 0].tolist() == [A, A] and (ids[:, 1] < A).all()


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
@pytest.mark.parametrize("kind", TEMPORAL)
def test_temporal_order_matters(kind, obs_kind):
    torch.manual_seed(0)
    m = make(kind, obs_kind, 2, False).eval()
    obs = random_obs(obs_kind, 2, False, batch=1)
    b = BASE[obs_kind]
    swapped = torch.cat([obs[:, b:], obs[:, :b]], dim=1)
    with torch.no_grad():
        assert not torch.allclose(m(obs).value, m(swapped).value)


def test_registry_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        ModelConfig(kind="gru")
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="unknown model kind"):
        build_policy(SimpleNamespace(kind="gru"), spec(), A)
    assert set(POLICY_KINDS) == {"transformer", "cnn", "resnet", "hybrid", "temporal_transformer", "frame_attention", "lstm", "mlp"}


def test_cnn_config_validation():
    with pytest.raises(ValueError, match="odd"):
        ModelConfig(kernel_size=4)
    with pytest.raises(ValueError):
        ModelConfig(conv_layers=0)


# Snake checkpoints: the state_dict keys and shapes of every grid kind, recorded before vector
# observations were added. A refactor that renames or reshapes any of them breaks every saved model.
GRID_KEYS = json.loads((Path(__file__).parent / "data" / "grid_state_dict_keys.json").read_text())
RECORDED_CONFIG = dict(d_model=16, num_heads=2, num_layers=2, mlp_ratio=2, conv_channels=8, conv_layers=2, kernel_size=3)
RECORDED_SPECS = {"k1": ObservationSpec("grid", (3, 5, 6), 1, 0), "k2_actions": ObservationSpec("grid", (14, 5, 6), 2, 4)}


@pytest.mark.parametrize("entry", sorted(GRID_KEYS))
def test_grid_state_dict_keys_and_shapes_never_change(entry):
    kind, variant = entry.split("/")
    m = build_policy(ModelConfig(kind=kind, **RECORDED_CONFIG), RECORDED_SPECS[variant], 4)
    assert [[key, list(t.shape)] for key, t in m.state_dict().items()] == [[key, shape] for key, shape in GRID_KEYS[entry].items()]


# --------------------------------------------------------------------------- resnet and lstm


def _zero_residual_branches(m):
    """Make every residual block's body output zero, leaving only its skip connection."""
    for name, p in m.named_parameters():
        if "blocks" in name and ("conv2" in name or "linear2" in name):
            torch.nn.init.zeros_(p)


@pytest.mark.parametrize("obs_kind, block", [("grid", "ResidualConvBlock"), ("vector", "ResidualMlpBlock")])
def test_resnet_diagram_shows_the_residual_blocks(obs_kind, block):
    text = diagram(make("resnet", obs_kind, 1, False, conv_layers=3), "text")
    assert block in text and ".." in text  # three identical blocks are collapsed into one row


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
def test_resnet_blocks_are_the_configured_number_of_residual_layers(obs_kind):
    m = make("resnet", obs_kind, 1, False, conv_layers=3)
    assert len(m.net["encoder"].blocks) == 3
    assert len(make("resnet", obs_kind, 1, False, conv_layers=1).net["encoder"].blocks) == 1


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
def test_resnet_gradient_reaches_the_input_through_the_skip_alone(obs_kind):
    """With every block body zeroed the stem still learns: the skip connections carry the gradient."""
    torch.manual_seed(0)
    m = make("resnet", obs_kind, 1, False, conv_layers=3)
    _zero_residual_branches(m)
    m(random_obs(obs_kind, 1, False, batch=4)).value.sum().backward()
    stem = m.net["encoder"].stem
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in stem.parameters())


def test_vector_residual_block_is_the_identity_when_its_body_is_zero():
    from attentive_snake.models.blocks.residual import ResidualMlpBlock

    block = ResidualMlpBlock(8)
    torch.nn.init.zeros_(block.linear2.weight)
    torch.nn.init.zeros_(block.linear2.bias)
    x = torch.randn(3, 8)
    assert torch.equal(block(x), x)


def test_grid_residual_block_adds_its_input_before_the_activation():
    from attentive_snake.models.blocks.residual import ResidualConvBlock

    block = ResidualConvBlock(4, 3)
    torch.nn.init.zeros_(block.conv2.weight)
    torch.nn.init.zeros_(block.conv2.bias)
    x = torch.randn(2, 4, H, W)
    assert torch.allclose(block(x), torch.nn.functional.gelu(x))


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
def test_lstm_gradient_reaches_the_oldest_frame(obs_kind):
    torch.manual_seed(0)
    m = make("lstm", obs_kind, 4, False)
    obs = random_obs(obs_kind, 4, False, batch=2).requires_grad_(True)
    m(obs).value.sum().backward()
    assert obs.grad[:, : BASE[obs_kind]].abs().sum() > 0  # frame 0 of the window


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
def test_lstm_depends_on_older_frames_and_not_on_earlier_calls(obs_kind):
    torch.manual_seed(0)
    m = make("lstm", obs_kind, 4, False).eval()
    obs = random_obs(obs_kind, 4, False, batch=1)
    changed = obs.clone()
    changed[:, : BASE[obs_kind]] += 1.0  # only the oldest of four frames
    with torch.no_grad():
        first = m(obs).value
        assert not torch.allclose(first, m(changed).value)
        assert torch.equal(first, m(obs).value)  # nothing is carried over between calls


def test_lstm_uses_num_layers_lstm_layers_of_d_model_size():
    lstm = make("lstm", "vector", 3, False, num_layers=2).net["recurrent"].lstm
    assert (lstm.num_layers, lstm.hidden_size, lstm.batch_first) == (2, SMALL["d_model"], True)


@pytest.mark.parametrize("obs_kind", OBSERVATIONS)
def test_lstm_diagram_shows_the_recurrent_block(obs_kind):
    assert "LSTM" in diagram(make("lstm", obs_kind, 4, True), "text")
