import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest
import torch

from attentive_snake.games.snake.types import Action, GameState
from attentive_snake.games.snake.view import board_view
from attentive_snake.views.board_window import BoardWindow
from attentive_snake.viz import attention_rollout, cls_attention_grid

H, W = 4, 5


def fake_maps(layers=2, batch=2, heads=3):
    t = 1 + H * W
    return [torch.rand(batch, heads, t, t).softmax(-1) for _ in range(layers)]


def test_cls_grid_shape_dtype_range():
    g = cls_attention_grid(fake_maps(), H, W)
    assert g.shape == (H, W) and g.dtype == np.float32
    assert g.max() == pytest.approx(1.0) and g.min() >= 0


def test_cls_grid_picks_cls_row_row_major():
    t = 1 + H * W
    a = torch.zeros(1, 1, t, t)
    a[0, 0, 0, 1 + 2 * W + 3] = 1.0  # row 2, col 3
    g = cls_attention_grid([a], H, W)
    assert g[2, 3] == 1.0 and g.sum() == 1.0


def test_layer_and_batch_selection():
    maps = fake_maps()
    assert np.array_equal(cls_attention_grid(maps, H, W, layer=0), cls_attention_grid(maps, H, W, layer=-2))
    assert not np.array_equal(
        cls_attention_grid(maps, H, W, batch_index=0), cls_attention_grid(maps, H, W, batch_index=1)
    )


def test_rollout_shape_range():
    g = attention_rollout(fake_maps(), H, W, batch_index=1)
    assert g.shape == (H, W) and g.dtype == np.float32
    assert g.max() == pytest.approx(1.0)


def test_renderer_heatmap_smoke():
    state = GameState(
        width=W, height=H, snake=((1, 1), (0, 1)), direction=Action.RIGHT,
        food=(3, 2), score=0, steps=0, steps_since_food=0, done=False,
    )
    r = BoardWindow(W, H, cell_size=10)
    try:
        r.render(board_view(state))
        r.render(board_view(state), np.random.rand(H, W).astype(np.float32))
        with pytest.raises(ValueError):
            r.render(board_view(state), np.zeros((H + 1, W)))
    finally:
        r.close()


def fake_temporal_maps(frames=3, actions=2, layers=2, batch=2, heads=3):
    t = 1 + frames * H * W + actions
    return [torch.rand(batch, heads, t, t).softmax(-1) for _ in range(layers)]


def test_cls_frames_shape_range_and_selection():
    from attentive_snake.viz import cls_attention_frames

    g = cls_attention_frames(fake_temporal_maps(), 3, H, W)
    assert g.shape == (3, H, W) and g.dtype == np.float32
    assert g.max() == pytest.approx(1.0) and g.min() >= 0


def test_cls_frames_picks_frame_and_cell():
    from attentive_snake.viz import cls_attention_frames

    t = 1 + 3 * H * W + 2
    a = torch.zeros(1, 1, t, t)
    a[0, 0, 0, 1 + 2 * H * W + 1 * W + 4] = 1.0  # frame 2, row 1, col 4
    g = cls_attention_frames([a], 3, H, W)
    assert g[2, 1, 4] == 1.0 and g.sum() == 1.0


def test_cls_frames_common_scale_across_frames():
    from attentive_snake.viz import cls_attention_frames

    t = 1 + 2 * H * W
    a = torch.zeros(1, 1, t, t)
    a[0, 0, 0, 1] = 0.8  # frame 0
    a[0, 0, 0, 1 + H * W] = 0.2  # frame 1
    g = cls_attention_frames([a], 2, H, W)
    assert g[0].max() == 1.0 and g[1].max() == pytest.approx(0.25)


def test_cls_over_frames_sums_to_one_and_reads_offset():
    from attentive_snake.viz import cls_attention_over_frames

    t = 1 + 4
    a = torch.zeros(1, 2, t, t)
    a[0, :, 0, 1:] = torch.tensor([1.0, 1.0, 2.0, 4.0])
    w = cls_attention_over_frames([a], 4)
    assert w.shape == (4,) and w.sum() == pytest.approx(1.0)
    assert w.tolist() == pytest.approx([0.125, 0.125, 0.25, 0.5])


def test_cls_elements_reads_a_slice_and_scales_to_max_one():
    from attentive_snake.viz import cls_attention_elements

    t = 1 + 6
    a = torch.zeros(1, 2, t, t)
    a[0, :, 0, 1:] = torch.tensor([9.0, 1.0, 2.0, 4.0, 0.0, 9.0])
    w = cls_attention_elements([a], start=2, count=4)
    assert w.dtype == np.float32 and w.tolist() == pytest.approx([0.25, 0.5, 1.0, 0.0])
