"""Two-player Atari: the titles and their sides, the match (seat planes, zero-sum rewards, ends, seeding), the
random-opponent env of evaluation and watching, its statistics, the config checks and self-play training."""

import dataclasses
import json

import numpy as np
import pytest

from attentive_snake.config import EnvConfig, ExperimentConfig, save_experiment
from attentive_snake.config.fork_rules import locked_fields
from attentive_snake.env.match_vec_env import MatchVecEnv
from attentive_snake.episode_metrics import EpisodeRecord
from attentive_snake.evaluation.player import play_games
from attentive_snake.evaluation.settings import EvalSettings
from attentive_snake.factory import make_env, make_policy, make_trainer, make_vec_env
from attentive_snake.games.atari_versus.config import VersusRules
from attentive_snake.games.atari_versus.console import Console
from attentive_snake.games.atari_versus.env import model_seat
from attentive_snake.games.atari_versus.game import GAME
from attentive_snake.games.atari_versus.metrics import versus_statistics
from attentive_snake.games.atari_versus.reward import VersusReward, versus_reward
from attentive_snake.games.atari_versus.titles import TITLE_KEYS, TITLES
from attentive_snake.games.registry import get_game
from attentive_snake.games.spec import GameSpec
from attentive_snake.games.validation import spec_problems
from attentive_snake.models.config import ModelConfig
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, TrainConfig
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.watch import WatchSession

SMALL = VersusRules(sticky_actions=0.0, screen_size=48, max_episode_steps=6)


def config_of(rules=SMALL, num_envs=2, algorithm="ppo", **env) -> ExperimentConfig:
    settings = PpoConfig(rollout_length=8, num_minibatches=2) if algorithm == "ppo" else DqnConfig(buffer_size=64, learning_starts=4, batch_size=4)
    return ExperimentConfig(
        game_name="atari_versus",
        game=rules,
        model=ModelConfig(kind="cnn", d_model=16, conv_channels=8),
        train=TrainConfig(total_env_steps=32, device="cpu", algorithm=algorithm),
        algorithm_settings=settings,
        env=EnvConfig(num_envs=num_envs, history_length=2, **env),
    )


def play(match, steps, seed, actions=(3, 4)):
    match.reset(seed)
    return [match.step(actions) for _ in range(steps)]


# --------------------------------------------------------------------------- the game and its titles


def test_registered_with_two_seats_and_a_self_play_match():
    assert get_game("atari_versus") is GAME
    assert (GAME.observation_kind, GAME.view_kind, GAME.seats) == ("image", "image", 2)
    assert spec_problems(GAME, "atari_versus") == []
    assert {"pong", "combat_tank", "combat_plane", "boxing"} <= set(TITLE_KEYS)


def test_a_spec_with_seats_needs_a_match_and_only_then():
    single = get_game("snake")
    assert "make_match" in " ".join(spec_problems(dataclasses.replace(single, seats=2), "snake"))
    assert "make_match" in " ".join(spec_problems(dataclasses.replace(single, make_match=GAME.make_match), "snake"))
    assert "seats" in " ".join(spec_problems(dataclasses.replace(single, seats=0), "snake"))
    assert isinstance(GAME, GameSpec)


@pytest.mark.parametrize("title", TITLES, ids=lambda title: title.key)
def test_every_title_is_a_two_player_mode_of_a_shipped_rom(title):
    console = Console(VersusRules(title=title.key))
    assert console._ale.numPlayersActive() == 2 and title.mode in list(console._ale.getAvailableModes(2))
    assert len(console.action_ids) == 18


@pytest.mark.parametrize(("key", "side"), [("pong", "right"), ("combat_tank", "left")])
def test_seat_0_is_the_player_on_the_side_the_table_names(key, side):
    """Move only seat 0: the pixels that change lie on the side its name in titles.py says."""
    rules = VersusRules(title=key, sticky_actions=0.0, frame_skip=1)
    still, moved = Console(rules), Console(rules)
    for console, action in ((still, 0), (moved, 4)):  # LEFT: turns the tank, moves the paddle
        console.start(1)
        for _ in range(100):
            console.hold([action, 0])
    _, columns = np.nonzero((still.rgb() != moved.rgb()).any(axis=2))
    middle = still.rgb().shape[1] / 2
    assert len(columns) and ((columns > middle).all() if side == "right" else (columns < middle).all())
    assert side in next(title for title in TITLES if title.key == key).players[0]


# --------------------------------------------------------------------------- the match


def test_each_seat_sees_the_same_picture_and_its_own_seat_plane():
    match = GAME.make_match(SMALL, VersusReward(), 0)
    frames = match.reset()
    assert frames.shape == (2, 2, 48, 48) and match.observation_shape == (2, 48, 48)
    assert np.array_equal(frames[0, 0], frames[1, 0]) and frames[0, 0].any()
    assert (frames[0, 1] == 0.0).all() and (frames[1, 1] == 1.0).all()
    step = match.step([0, 0])
    assert (step[0].observation[1] == 0.0).all() and (step[1].observation[1] == 1.0).all()
    colour = GAME.make_match(dataclasses.replace(SMALL, grayscale=False), VersusReward(), 0)
    assert colour.reset().shape == (2, 4, 48, 48)


def test_rewards_and_scores_are_zero_sum_and_both_seats_end_together():
    rules = dataclasses.replace(SMALL, max_episode_steps=27000)
    match = GAME.make_match(rules, VersusReward(clip=False), 3)
    match.reset()
    rng = np.random.default_rng(0)
    events = set()
    while True:
        first, second = match.step(rng.integers(0, 18, size=2))
        assert first.reward == -second.reward and first.score == -second.score
        assert (first.terminated, first.truncated) == (second.terminated, second.truncated)
        events |= {first.event, second.event}
        if first.terminated:
            break
    assert {"scored", "conceded"} <= events
    assert {first.event, second.event} in ({"won", "lost"}, {"draw"})
    assert (first.event == "won") == (first.score > 0)


def test_the_step_limit_truncates_both_seats():
    steps = play(GAME.make_match(SMALL, VersusReward(), 0), 6, 0)
    assert all(not step[0].truncated for step in steps[:-1])
    assert [(seat.truncated, seat.terminated, seat.event) for seat in steps[-1]] == [(True, False, "timeout")] * 2


def test_a_seed_replays_the_same_match_and_another_seed_does_not():
    rules = dataclasses.replace(SMALL, sticky_actions=0.25, max_episode_steps=27000)

    def trace(seed):
        return [(seat.reward, seat.observation.tobytes()) for step in play(GAME.make_match(rules, VersusReward(), None), 60, seed) for seat in step]

    assert trace(5) == trace(5)
    assert trace(5) != trace(6)


def test_the_reward_clips_and_scales():
    assert versus_reward(VersusReward(), 2.0) == 2.0  # real points by default
    assert versus_reward(VersusReward(clip=True), 2.0) == 1.0 and versus_reward(VersusReward(clip=True), -1.0) == -1.0
    assert versus_reward(VersusReward(scale=0.5), 2.0) == 1.0
    with pytest.raises(ValueError):
        VersusReward(scale=0.0)


# --------------------------------------------------------------------------- a scripted Boxing match, slot by slot


BOXING = VersusRules(title="boxing", sticky_actions=0.0)
NOOP, RIGHT, LEFT = 0, 3, 4


def boxing_slots(history_length=2):
    vec = MatchVecEnv([GAME.make_match(BOXING, VersusReward(clip=True), 0)], history_length=history_length)
    return vec, vec.reset(seed=0)


@pytest.mark.parametrize(("seat", "script_seed", "buttons"), [(0, 0, (3, 5, 8, 1, 11, 13, 16)), (1, 1, (2, 4, 7, 1, 10, 12, 15))])
def test_a_boxing_slots_first_points_go_to_the_seat_that_punched(seat, script_seed, buttons):
    """Only ``seat`` plays (walking at the idle other boxer and punching): its slot gets +1, the other slot -1."""
    vec, _ = boxing_slots()
    rng = np.random.default_rng(script_seed)
    for _ in range(100):
        actions = np.zeros(2, dtype=np.int64)
        actions[seat] = rng.choice(buttons)
        rewards = vec.step(actions).rewards
        if rewards.any():
            break
    assert rewards[seat] == 1.0 and rewards[1 - seat] == -1.0


def test_a_boxing_slots_action_moves_its_own_boxer_only():
    """The white boxer (seat 0) starts in the top half of the ring, the black one (seat 1) in the bottom half."""

    def picture_after(actions):
        vec, _ = boxing_slots(history_length=1)
        for _ in range(6):
            observation = vec.step(np.array(actions)).observations
        return observation[0, 0]

    still = picture_after([NOOP, NOOP])
    rows_of_seat_0 = np.nonzero(picture_after([RIGHT, NOOP]) != still)[0]
    rows_of_seat_1 = np.nonzero(picture_after([NOOP, LEFT]) != still)[0]
    assert len(rows_of_seat_0) and rows_of_seat_0.max() < 42
    assert len(rows_of_seat_1) and rows_of_seat_1.min() >= 42


def test_a_boxing_slots_observations_share_the_picture_differ_by_seat_and_change_over_time():
    vec, first = boxing_slots()
    assert (first[0, 3] == 0.0).all() and (first[1, 3] == 1.0).all()  # newest frame: picture, seat plane
    rng, pictures = np.random.default_rng(0), [first[0, 2].copy()]
    for _ in range(20):
        observations = vec.step(rng.integers(18, size=2)).observations
        assert np.array_equal(observations[0, [0, 2]], observations[1, [0, 2]])
        assert (observations[0, [1, 3]] == 0.0).all() and (observations[1, [1, 3]] == 1.0).all()
        assert np.array_equal(observations[0, 0], pictures[-1])  # the older frame is the previous newest
        pictures.append(observations[0, 2].copy())
    assert all(not np.array_equal(before, after) for before, after in zip(pictures[4:], pictures[5:]))


def test_a_new_boxing_match_forgets_the_last_ones_frames():
    vec, _ = boxing_slots()
    rng = np.random.default_rng(0)
    while not (step := vec.step(rng.integers(18, size=2))).terminated.any():
        pass
    assert sorted(step.final_observations) == [0, 1] and step.final_observations[0][0].any()
    assert (step.observations[:, :2] == 0.0).all()  # no frame before the first one of the new match
    assert (step.observations[0, 3] == 0.0).all() and (step.observations[1, 3] == 1.0).all()


# --------------------------------------------------------------------------- against the random player


def test_the_model_takes_the_seed_parity_seat_and_alternates_without_a_seed():
    assert [model_seat(seed, 0) for seed in (4, 7)] == [0, 1]
    env = GAME.make_env(SMALL, VersusReward(), None)
    seats = []
    for _ in range(3):
        env.reset()
        seats.append(env.seat)
    assert seats == [0, 1, 0]
    env.reset(seed=11)
    assert env.seat == 1 and "orange paddle on the left" in env.view()["note"]


def test_the_random_player_is_seeded_so_a_game_replays():
    def trace(seed):
        env = GAME.make_env(dataclasses.replace(SMALL, max_episode_steps=200), VersusReward(), None)
        first = env.reset(seed)
        return first.tobytes(), [(step.reward, step.observation.tobytes()) for step in (env.step(0) for _ in range(40))]

    assert trace(3) == trace(3)
    assert trace(3) != trace(5)


def test_the_env_plays_the_model_seat_of_the_match():
    env = GAME.make_env(SMALL, VersusReward(), 0)
    observation = env.reset(seed=1)
    assert env.seat == 1 and (observation[1] == 1.0).all()
    step = env.step(2)
    assert (step.observation[1] == 1.0).all() and step.event in ("playing", "scored", "conceded")


def test_win_draw_and_loss_rates_count_scores():
    def record(score, outcome="won"):
        return EpisodeRecord(score, 10, score, outcome, ("playing",) * 9 + (outcome,))

    stats = versus_statistics([record(3.0), record(-1.0, "lost"), record(0.0, "timeout"), record(2.0, "timeout")])
    assert stats == {"win_rate": 0.5, "draw_rate": 0.25, "loss_rate": 0.25, "timeout_rate": 0.5}


def test_an_evaluation_against_the_random_player_is_deterministic():
    config = config_of()
    policy = make_policy(config, make_env(config))
    settings = EvalSettings(games=3, seed=1000, max_episode_steps=100, parallel_games=2)
    first = play_games(policy, lambda: make_env(config), settings)
    assert first == play_games(policy, lambda: make_env(config), settings)
    assert len(first) == 3 and all(record.outcome == "timeout" for record in first)


# --------------------------------------------------------------------------- config, training, watching


def test_num_envs_must_hold_whole_matches():
    with pytest.raises(ValueError, match="multiple of 2"):
        config_of(num_envs=3)
    config_of(num_envs=4)


def test_the_title_and_the_picture_are_locked_on_fork():
    assert {"game.title", "game.screen_size", "game.grayscale", "game.full_action_space"} <= set(locked_fields("atari_versus"))


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_self_play_trains_on_every_seat_of_the_matches(algorithm):
    config = config_of(num_envs=4, algorithm=algorithm, history_actions=True)
    env = make_vec_env(config)
    assert isinstance(env, MatchVecEnv) and len(env.matches) == 2 and env.num_envs == 4
    assert env.observation_shape == make_env(config).observation_shape == (2 * (2 + 18), 48, 48)
    trainer = make_trainer(config, make_policy(config, env), env, NullLogger())
    trainer.train()
    assert trainer.env_steps >= 32


def test_watching_shows_the_match_and_names_the_model_player(tmp_path):
    config = dataclasses.replace(config_of(), train=TrainConfig(total_env_steps=16, checkpoint_dir=tmp_path, device="cpu"))
    save_experiment(config, config_path(tmp_path))
    env = make_vec_env(config)
    make_trainer(config, make_policy(config, env), env, NullLogger()).train()
    session = WatchSession(tmp_path, seed=0)
    assert json.loads(json.dumps(session.info)) == session.info
    assert (session.info["view_kind"], session.info["width"], session.info["height"]) == ("image", 160, 210)
    assert session.info["observation_shape"] == [4, 48, 48] and len(session.info["action_names"]) == 18
    frame = session.current().to_dict()
    assert frame["note"].startswith("The model plays the ") and frame["png"]
    assert len(frame["action_values"]["values"]) == 18
