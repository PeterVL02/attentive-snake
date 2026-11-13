import numpy as np
import pytest

from attentive_snake.config.experiment import EnvConfig
from attentive_snake.env.frame import FrameBuilder
from attentive_snake.env.history import FrameHistory
from attentive_snake.env.observation import ObservationEncoder
from attentive_snake.env.snake_env import SnakeEnv
from attentive_snake.env.vec_env import SyncVecSnakeEnv
from attentive_snake.game.config import GameConfig
from attentive_snake.game.types import Action

CFG = GameConfig(width=7, height=6)


def frames(obs, k, f):
    return [obs[i * f : (i + 1) * f] for i in range(k)]


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("actions", [False, True])
def test_shapes(k, actions):
    f = 3 + 4 * actions
    env = SnakeEnv(CFG, seed=0, history_length=k, history_actions=actions)
    assert env.observation_shape == (k * f, 6, 7)
    assert env.history_length == k and env.frame_channels == f
    assert env.reset().shape == env.observation_shape
    step = env.step(env.state.direction)
    assert step.observation.shape == env.observation_shape
    assert step.observation.dtype == np.float32
    venv = SyncVecSnakeEnv(3, CFG, seed=0, history_length=k, history_actions=actions)
    assert venv.observation_shape == (k * f, 6, 7)
    assert venv.history_length == k and venv.frame_channels == f
    assert venv.reset().shape == (3, k * f, 6, 7)


def test_default_is_plain_encoding():
    env = SnakeEnv(CFG, seed=5)
    enc = ObservationEncoder()
    np.testing.assert_array_equal(env.reset(), enc.encode(env.state))
    for _ in range(5):
        step = env.step(env.state.direction)
        np.testing.assert_array_equal(step.observation, enc.encode(env.state))
        if step.terminated or step.truncated:
            break


def test_oldest_first_and_zero_padding():
    env = SnakeEnv(CFG, seed=1, history_length=3)
    enc = ObservationEncoder()
    obs = env.reset()
    s0 = enc.encode(env.state)
    a, b, c = frames(obs, 3, 3)
    assert not a.any() and not b.any()
    np.testing.assert_array_equal(c, s0)
    obs = env.step(env.state.direction).observation
    s1 = enc.encode(env.state)
    a, b, c = frames(obs, 3, 3)
    assert not a.any()
    np.testing.assert_array_equal(b, s0)
    np.testing.assert_array_equal(c, s1)
    obs = env.step(env.state.direction).observation
    s2 = enc.encode(env.state)
    a, b, c = frames(obs, 3, 3)
    np.testing.assert_array_equal(a, s0)
    np.testing.assert_array_equal(b, s1)
    np.testing.assert_array_equal(c, s2)
    obs = env.step(env.state.direction).observation
    np.testing.assert_array_equal(frames(obs, 3, 3)[0], s1)  # s0 dropped


def planes(frame):
    return frame[3:, 0, 0]


def test_action_planes_record_chosen_action():
    env = SnakeEnv(CFG, seed=1, history_length=2, history_actions=True)
    obs = env.reset()
    assert not planes(frames(obs, 2, 7)[1]).any()  # reset frame: no action
    direction = env.state.direction
    opposite = {Action.UP: Action.DOWN, Action.DOWN: Action.UP,
                Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT}[direction]
    obs = env.step(direction).observation
    f = frames(obs, 2, 7)
    assert not planes(f[0]).any()
    assert planes(f[1]).tolist() == np.eye(4)[int(direction)].tolist()
    assert (f[1][3:] == f[1][3:, :1, :1]).all()  # constant planes
    # Reverse action is ignored by the engine, but the chosen one is recorded.
    before = env.state.direction
    obs = env.step(opposite).observation
    assert env.state.direction == before
    f = frames(obs, 2, 7)
    assert planes(f[0]).tolist() == np.eye(4)[int(direction)].tolist()
    assert planes(f[1]).tolist() == np.eye(4)[int(opposite)].tolist()


def test_history_cleared_on_reset():
    env = SnakeEnv(CFG, seed=2, history_length=3, history_actions=True)
    env.reset()
    for _ in range(3):
        env.step(env.state.direction)
    obs = env.reset()
    a, b, c = frames(obs, 3, 7)
    assert not a.any() and not b.any() and c[:3].any() and not c[3:].any()


def test_vec_auto_reset_starts_clean_history():
    cfg = GameConfig(width=5, height=5)
    venv = SyncVecSnakeEnv(2, cfg, seed=0, history_length=3, history_actions=True)
    venv.reset()
    for _ in range(200):
        step = venv.step(np.array([0, 1]))
        done = step.terminated | step.truncated
        if done.any():
            i = int(np.argmax(done))
            a, b, c = frames(step.observations[i], 3, 7)
            assert not a.any() and not b.any() and not c[3:].any() and c[:3].any()
            return
    pytest.fail("no episode finished")


def test_no_aliasing():
    env = SnakeEnv(CFG, seed=3, history_length=3, history_actions=True)
    first = env.reset()
    snap = first.copy()
    s1 = env.step(env.state.direction).observation
    snap1 = s1.copy()
    env.step(env.state.direction)
    np.testing.assert_array_equal(first, snap)
    np.testing.assert_array_equal(s1, snap1)
    venv = SyncVecSnakeEnv(2, CFG, seed=0, history_length=3)
    o0 = venv.reset()
    keep = o0.copy()
    venv.step(np.array([0, 0]))
    np.testing.assert_array_equal(o0, keep)


def test_frame_builder_and_history_units():
    env = SnakeEnv(CFG, seed=0)
    builder = FrameBuilder(ObservationEncoder(), with_actions=True)
    f = builder.build(env.state, Action.LEFT)
    assert f.shape == builder.shape(7, 6) == (7, 6, 7)
    assert f[3 + int(Action.LEFT)].all() and f[3:].sum() == 42
    h = FrameHistory(2, (1, 2, 2))
    h.push(np.ones((1, 2, 2), np.float32))
    out = h.push(np.full((1, 2, 2), 2, np.float32))
    assert out[0].max() == 1 and out[1].max() == 2
    with pytest.raises(ValueError):
        FrameHistory(0, (1, 2, 2))


def test_env_config_validation():
    assert EnvConfig().history_length == 1 and EnvConfig().history_actions is False
    with pytest.raises(ValueError):
        EnvConfig(history_length=0)
