"""Image observations: the strided-conv and patch stages, and every model kind that reads pictures."""

import dataclasses

import pytest
import torch
from torch import nn

from attentive_snake.config import EnvConfig, ExperimentConfig
from attentive_snake.factory import make_env, make_policy
from attentive_snake.games.atari.config import AtariRules
from attentive_snake.games.atari.reward import AtariReward
from attentive_snake.games.atari.titles import TITLE_KEYS
from attentive_snake.models.blocks.image import DownsamplingConvEncoder, PatchEmbedding, downsampled_size, patch_count
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.heads import HEADS
from attentive_snake.models.input_stages import image
from attentive_snake.models.input_stages.catalog import FEATURE_STAGES, FRAME_STAGES, HYBRID_STAGES, RESIDUAL_STAGES, TOKEN_STAGES
from attentive_snake.models.input_stages.stage import StageInput
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.observation_spec import ObservationSpec

IMAGE_KINDS = [kind for kind in POLICY_KINDS if "image" in KIND_INFO[kind].inputs]
ACTIONS = 18
SMALL = ModelConfig(d_model=16, num_heads=2, num_layers=1, mlp_ratio=1, conv_channels=8, conv_layers=1)


def spec(kind: str, size: int = 48, actions: bool = False) -> ObservationSpec:
    """Grayscale pictures; temporal kinds get 2 frames (with 18 action planes each if ``actions``)."""
    frames = 2 if KIND_INFO[kind].uses_history else 1
    channels = 1 + (ACTIONS if actions else 0)
    return ObservationSpec("image", (frames * channels, size, size), frames, action_channels=ACTIONS if actions else 0)


def build(kind: str, head: str = "actor_critic", size: int = 48, actions: bool = False, **changes) -> nn.Module:
    return build_policy(dataclasses.replace(SMALL, kind=kind, **changes), spec(kind, size, actions), ACTIONS, head)


def parameters(policy: nn.Module) -> int:
    return sum(p.numel() for p in policy.parameters())


# --------------------------------------------------------------------------- blocks and stages


def test_the_conv_encoder_is_the_dqn_papers_and_shrinks_84_to_7():
    encoder = DownsamplingConvEncoder(4, 64)
    convs = [layer for layer in encoder.layers if isinstance(layer, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.stride) for c in convs] == [
        (4, 32, (8, 8), (4, 4)), (32, 64, (4, 4), (2, 2)), (64, 64, (3, 3), (1, 1)),
    ]  # fmt: skip
    assert encoder(torch.rand(2, 4, 84, 84)).shape == (2, 64, 7, 7)
    assert (downsampled_size(84), downsampled_size(48), downsampled_size(36)) == (7, 2, 1)
    with pytest.raises(ValueError, match="too small"):
        downsampled_size(35)
    normed = DownsamplingConvEncoder(4, 64, norm=True)
    assert sum(isinstance(layer, nn.GroupNorm) for layer in normed.layers) == 3 and not any(isinstance(layer, nn.GroupNorm) for layer in encoder.layers)


def test_patches_become_tokens_row_by_row():
    embed = PatchEmbedding(1, 8, patch=2)
    with torch.no_grad():
        embed.proj.weight.fill_(1.0)
        embed.proj.bias.zero_()
    picture = torch.zeros(1, 1, 4, 6)
    picture[0, 0, 0, 4] = 1.0  # in the top-right patch: row 0, column 2
    tokens = embed(picture)
    assert tokens.shape == (1, 6, 8) and tokens[0, :, 0].tolist() == [0, 0, 1, 0, 0, 0]
    assert (patch_count(84, 12), patch_count(50, 12)) == (7, 4)
    with pytest.raises(ValueError, match="smaller than one"):
        patch_count(8, 12)


def test_stages_say_how_many_tokens_and_in_what_grid():
    source, config = StageInput(4, 1, (84, 84)), dataclasses.replace(SMALL, kind="cnn")
    tokens, hybrid = image.token_stage(source, config), image.hybrid_stage(source, config)
    assert (tokens.num_tokens, tokens.token_grid) == (49, (7, 7)) == (hybrid.num_tokens, hybrid.token_grid)
    assert list(tokens.modules) == ["embed", "positional"] and list(hybrid.modules) == ["stem", "project", "tokens", "positional"]
    feature = image.feature_stage(source, config)
    assert list(feature.modules) == ["encoder", "readout"] and (feature.num_tokens, feature.token_grid) == (0, None)
    for table in (TOKEN_STAGES, FEATURE_STAGES, FRAME_STAGES, HYBRID_STAGES):
        assert set(table) == {"grid", "vector", "image"}
    assert set(RESIDUAL_STAGES) == {"grid", "vector"}  # no residual image stage: resnet does not read pictures


# --------------------------------------------------------------------------- every kind on pictures


def test_the_kinds_that_read_pictures():
    assert IMAGE_KINDS == ["transformer", "cnn", "hybrid", "temporal_transformer", "frame_attention", "lstm", "jepa", "jepa_attention", "jepa_tokens", "jepa_temporal"]
    for kind in ("resnet", "mlp"):
        with pytest.raises(ValueError, match="cannot read image observations"):
            build(kind)


@pytest.mark.parametrize("head", list(HEADS))
@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_builds_and_runs_with_every_head(kind, head):
    if head == "gaussian" and KIND_INFO[kind].needs_discrete_actions:  # the world models have no continuous head
        with pytest.raises(ValueError, match="unknown policy head 'gaussian'"):
            build(kind, head)
        return
    policy = build(kind, head)
    assert policy.observation_kinds == KIND_INFO[kind].observation_kinds
    out = policy(torch.rand(3, *policy.observation_shape))
    values = out.mean if hasattr(out, "mean") else out.logits if hasattr(out, "logits") else out.q_values
    assert values.shape == (3, ACTIONS) and torch.isfinite(values).all()
    extra = (out.value.sum() if hasattr(out, "value") else 0.0) + (out.log_std.sum() if hasattr(out, "log_std") else 0.0)
    (values.sum() + extra).backward()
    assert all(p.grad is not None for p in policy.parameters())  # every weight takes part


@pytest.mark.parametrize("kind", ["temporal_transformer", "frame_attention", "lstm", "jepa_temporal"])
def test_temporal_kinds_read_the_action_planes(kind):
    policy = build(kind, actions=True)
    assert "action_embed" in policy.net and policy.observation_shape == (2 * 19, 48, 48)
    assert policy(torch.rand(2, *policy.observation_shape)).logits.shape == (2, ACTIONS)


@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_attention_maps_cover_the_token_grid(kind):
    policy = build(kind, size=84)
    view = KIND_INFO[kind].attention_view
    if view is None:
        assert not hasattr(policy, "attention_maps")
        return
    layout = policy.token_layout
    maps = policy.attention_maps(torch.rand(1, *policy.observation_shape))
    assert maps[0].shape == (1, 2, layout.num_tokens, layout.num_tokens)
    if view == "weights":
        assert (layout.tokens_per_frame, layout.grid, layout.num_tokens) == (1, None, 3)
    else:
        assert (layout.tokens_per_frame, layout.grid) == (49, (7, 7)) and layout.num_frames == policy.history_length


# Fields that change the parameter count when used (as in tests/test_schema.py, here on pictures).
SIZE_CHANGES = {
    "d_model": 32,
    "num_layers": 2,
    "predictor_layers": 2,
    "mlp_ratio": 2,
    "conv_channels": 24,
    "conv_layers": 2,
    "kernel_size": 5,
    "residual": True,
    "layer_norm": True,
}


@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_size_fields_change_the_model_exactly_when_listed_for_pictures(kind):
    reference = parameters(build(kind, actions=KIND_INFO[kind].uses_history))
    listed = KIND_INFO[kind].fields_for("image")
    for name, value in SIZE_CHANGES.items():
        changed = parameters(build(kind, actions=KIND_INFO[kind].uses_history, **{name: value}))
        assert (changed != reference) == (name in listed), f"{kind} on image: {name}"


def test_readings_name_only_listed_fields_and_grid_and_vector_readings_are_untouched():
    for kind in IMAGE_KINDS:
        info = KIND_INFO[kind]
        assert set(info.inputs["image"].unused_fields) <= set(info.fields) and info.inputs["image"].description.strip()
        assert list(info.inputs)[:2] == ["grid", "vector"]


def test_the_default_cnn_is_the_dqn_papers_network():
    policy = build_policy(ModelConfig(kind="cnn", d_model=512, conv_channels=64), ObservationSpec("image", (4, 84, 84), 4), ACTIONS)
    assert policy.net["readout"].linear.in_features == 64 * 7 * 7 and policy.net["readout"].linear.out_features == 512
    assert parameters(policy) == 1_693_875  # 78k in the convolutions, 1.6M in the readout
    assert list(policy.state_dict()) == [  # checkpoint keys of image cnn runs: renaming a module breaks them
        "net.encoder.layers.0.weight", "net.encoder.layers.0.bias", "net.encoder.layers.2.weight", "net.encoder.layers.2.bias",
        "net.encoder.layers.4.weight", "net.encoder.layers.4.bias", "net.readout.linear.weight", "net.readout.linear.bias",
        "net.policy_head.linear.weight", "net.policy_head.linear.bias", "net.value_head.linear.weight", "net.value_head.linear.bias",
    ]  # fmt: skip


def test_with_the_full_action_space_every_title_gets_the_same_network():
    """Same weights' shapes whatever the title: 18 actions and one picture size everywhere."""
    shapes = set()
    for title in TITLE_KEYS:
        config = ExperimentConfig(
            game_name="atari", game=AtariRules(title=title), reward=AtariReward(),
            model=dataclasses.replace(SMALL, kind="cnn"), env=EnvConfig(history_length=4),
        )  # fmt: skip
        policy = make_policy(config, make_env(config))
        shapes.add(tuple((name, tuple(value.shape)) for name, value in policy.state_dict().items()))
    assert len(shapes) == 1
    own = ExperimentConfig(game_name="atari", game=AtariRules(full_action_space=False), reward=AtariReward(), model=dataclasses.replace(SMALL, kind="cnn"))
    assert make_policy(own, make_env(own)).num_actions == 6
