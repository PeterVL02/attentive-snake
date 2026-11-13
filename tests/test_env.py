import numpy as np
import pytest

from attentive_snake.env.observation import ObservationEncoder
from attentive_snake.env.reward import RewardConfig, build_reward
from attentive_snake.env.snake_env import SnakeEnv
from attentive_snake.env.vec_env import SyncVecSnakeEnv
from attentive_snake.game.config import GameConfig
from attentive_snake.game.types import Action, GameState, StepEvent, StepResult


def make_state(snake, food, w=6, h=5, done=False):
    return GameState(w, h, tuple(snake), Action.RIGHT, food, 0, 0, 0, done)


def result(state, event):
    return StepResult(state, event)


CFG = RewardConfig(food=2.0, death=-3.0, win=5.0, starved=-4.0, step=-0.1,
                   closer_to_food=0.3, farther_from_food=-0.2)


def test_reward_events():
    f = build_reward(CFG)
    prev = make_state([(2, 2), (1, 2)], (5, 2))
    new = make_state([(3, 2), (2, 2)], (5, 2))
    assert f(prev, result(new, StepEvent.ATE_FOOD)) == pytest.approx(1.9)
    for ev, val in [(StepEvent.HIT_WALL, -3.0), (StepEvent.HIT_SELF, -3.0),
                    (StepEvent.WON, 5.0), (StepEvent.STARVED, -4.0)]:
        assert f(prev, result(new, ev)) == val


def test_reward_shaping():
    f = build_reward(CFG)
    prev = make_state([(2, 2), (1, 2)], (5, 2))
    closer = make_state([(3, 2), (2, 2)], (5, 2))
    farther = make_state([(1, 2), (2, 2)], (5, 2))
    same = make_state([(2, 3), (2, 2)], (5, 2))  # distance 3 -> 4 would be farther
    assert f(prev, result(closer, StepEvent.MOVED)) == pytest.approx(-0.1 + 0.3)
    assert f(prev, result(farther, StepEvent.MOVED)) == pytest.approx(-0.1 - 0.2)
    equal = make_state([(3, 3), (2, 3)], (5, 2))  # dist 3 -> 3
    assert f(prev, result(equal, StepEvent.MOVED)) == pytest.approx(-0.1)
    assert same is not None


def test_reward_defaults_no_shaping():
    f = build_reward(RewardConfig())
    prev = make_state([(2, 2), (1, 2)], (5, 2))
    new = make_state([(3, 2), (2, 2)], (5, 2))
    assert f(prev, result(new, StepEvent.MOVED)) == pytest.approx(-0.01)


def test_env_accepts_custom_reward():
    env = SnakeEnv(GameConfig(width=8, height=6), lambda prev, res: 7.0, seed=1)
    env.reset()
    assert env.step(env.state.direction).reward == 7.0
    venv = SyncVecSnakeEnv(2, GameConfig(), RewardConfig(step=-0.5), seed=0)
    venv.reset()
    assert venv.step(np.array([1, 1])).rewards[0] <= 1.0


def test_observation_indexing_non_square():
    enc = ObservationEncoder()
    assert enc.shape(6, 5) == (3, 5, 6)
    state = make_state([(3, 1), (2, 1), (1, 1)], (5, 4))
    obs = enc.encode(state)
    assert obs.shape == (3, 5, 6) and obs.dtype == np.float32
    assert obs[0, 1, 3] == 1.0 and obs[0].sum() == 1.0
    assert obs[1, 1, 3] == 1.0
    assert obs[1, 1, 2] == pytest.approx(2 / 3)
    assert obs[1, 1, 1] == pytest.approx(1 / 3)
    assert np.count_nonzero(obs[1]) == 3
    assert obs[2, 4, 5] == 1.0 and obs[2].sum() == 1.0


def test_observation_no_food():
    obs = ObservationEncoder().encode(make_state([(0, 0)], None))
    assert obs[2].sum() == 0.0


def test_env_contract():
    env = SnakeEnv(GameConfig(width=8, height=6), seed=1)
    assert env.observation_shape == (3, 6, 8) and env.num_actions == 4
    obs = env.reset()
    assert obs.shape == env.observation_shape and obs.dtype == np.float32
    s = env.step(env.state.direction)
    assert s.observation.shape == env.observation_shape
    assert isinstance(s.reward, float)
    assert s.event in (StepEvent.MOVED, StepEvent.ATE_FOOD)
    assert not s.terminated and not s.truncated


def test_env_wall_terminates_and_starvation_truncates():
    env = SnakeEnv(GameConfig(width=5, height=5), seed=0)
    env.reset()
    for _ in range(10):
        s = env.step(Action.UP)
        if s.terminated:
            break
    assert s.event.is_terminal and env.state.done
    env = SnakeEnv(GameConfig(width=20, height=20, max_steps_without_food=2), seed=0)
    env.reset()
    env.step(env.state.direction)
    s = env.step(env.state.direction)
    assert s.truncated and not s.terminated and s.event is StepEvent.STARVED


def run_vec(seed, steps=300, n=8):
    venv = SyncVecSnakeEnv(n, GameConfig(width=6, height=5, max_steps_without_food=40), seed=seed)
    rng = np.random.default_rng(123)
    obs = [venv.reset()]
    out = []
    for _ in range(steps):
        r = venv.step(rng.integers(0, 4, size=n))
        out.append(r)
        obs.append(r.observations)
    return venv, obs, out


def test_vec_shapes_dtypes():
    venv, _, out = run_vec(0, steps=5)
    r = out[0]
    assert venv.observation_shape == (3, 5, 6) and venv.num_actions == 4
    assert r.observations.shape == (8, 3, 5, 6) and r.observations.dtype == np.float32
    assert r.rewards.shape == (8,) and r.rewards.dtype == np.float32
    assert r.terminated.dtype == bool and r.truncated.dtype == bool
    assert len(venv.states()) == 8


def test_vec_auto_reset_and_summaries():
    venv = SyncVecSnakeEnv(4, GameConfig(width=5, height=5), seed=3)
    venv.reset()
    returns = np.zeros(4)
    lengths = np.zeros(4, int)
    seen = 0
    enc = ObservationEncoder()
    for _ in range(40):
        r = venv.step(np.full(4, Action.UP))
        returns += r.rewards
        lengths += 1
        for ep in r.finished_episodes:
            i = ep.env_index
            seen += 1
            assert r.terminated[i] and ep.final_event.is_terminal
            assert ep.total_reward == pytest.approx(returns[i], abs=1e-5)
            assert ep.length == lengths[i]
            returns[i] = 0
            lengths[i] = 0
            # obs is the fresh episode's
            fresh = venv.states()[i]
            assert not fresh.done and fresh.steps == 0
            np.testing.assert_array_equal(r.observations[i], enc.encode(fresh))
    assert seen >= 4


def test_vec_determinism_and_divergence():
    _, obs_a, out_a = run_vec(7)
    _, obs_b, out_b = run_vec(7)
    for a, b in zip(obs_a, obs_b):
        np.testing.assert_array_equal(a, b)
    assert [e.score for r in out_a for e in r.finished_episodes] == \
           [e.score for r in out_b for e in r.finished_episodes]
    # sub-envs differ from each other at reset
    first = obs_a[0]
    assert any(not np.array_equal(first[0], first[i]) for i in range(1, 8))
    _, obs_c, _ = run_vec(8)
    assert not np.array_equal(obs_a[0], obs_c[0])


def test_vec_obs_not_aliased():
    venv = SyncVecSnakeEnv(2, GameConfig(), seed=0)
    o = venv.reset()
    r = venv.step(np.array([1, 1]))
    assert not np.shares_memory(o, r.observations)
