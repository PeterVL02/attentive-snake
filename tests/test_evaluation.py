"""Evaluation: statistics (generic and per game), greedy batched play, the training hook, records and reproduction."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.env.types import EnvStep
from attentive_snake.env.vec_env import env_seed
from attentive_snake.episode_metrics import STEP_LIMIT, EpisodeRecord, GameMetrics, MetricDefinition, median
from attentive_snake.evaluation.config import EvalConfig
from attentive_snake.evaluation.generic_statistics import GENERIC_METRICS, generic_statistics, outcome_counts
from attentive_snake.evaluation.hook import PeriodicEvaluation
from attentive_snake.evaluation.log import append_evaluation, read_evaluations
from attentive_snake.evaluation.matching import find_reproduction
from attentive_snake.evaluation.player import play_games
from attentive_snake.evaluation.record import Evaluation
from attentive_snake.evaluation.schedule import evaluation_due
from attentive_snake.evaluation.settings import EvalSettings, settings_from_config
from attentive_snake.evaluation.statistics import evaluation_statistics, metric_definitions
from attentive_snake.evaluation.table import evaluation_table, format_value
from attentive_snake.factory import make_env, make_policy, make_vec_env, make_trainer
from attentive_snake.games.lunar_lander.metrics import lander_statistics
from attentive_snake.games.registry import get_game
from attentive_snake.games.snake.metrics import apple_intervals, snake_statistics
from attentive_snake.games.validation import metric_problems
from attentive_snake.models.types import PolicyOutput
from attentive_snake.training import NullLogger, Trainer


def record(outcome: str, events: list[str], score: float = 0.0, total_reward: float = 0.0) -> EpisodeRecord:
    return EpisodeRecord(score, len(events), total_reward, outcome, tuple(events))


def snake_game(*events: str) -> EpisodeRecord:
    """A Snake record from compact events: ``m`` moved, ``a`` ate food, ``w`` won, else the outcome."""
    names = {"m": "moved", "a": "ate_food", "w": "won"}
    full = [names.get(event, event) for event in events]
    return record(full[-1], full, score=sum(e in ("ate_food", "won") for e in full))


# --------------------------------------------------------------------------- statistics


def test_median_and_generic_statistics() -> None:
    assert median([3, 1, 2]) == 2 and median([4, 1, 2, 3]) == 2.5 and median([]) is None
    records = [record("a", ["x"] * 4, score=2, total_reward=1.0), record("b", ["x"] * 2, score=6, total_reward=-1.0), record("a", ["x"], score=1)]
    stats = generic_statistics(records)
    assert stats == {
        "games": 3.0, "mean_score": 3.0, "median_score": 2.0, "min_score": 1.0, "max_score": 6.0,
        "mean_length": 7 / 3, "mean_return": 0.0,
    }
    assert list(stats) == [definition.key for definition in GENERIC_METRICS]
    assert outcome_counts(records) == {"a": 2, "b": 1}
    empty = generic_statistics([])
    assert empty["games"] == 0 and all(empty[key] is None for key in empty if key != "games")  # N/A, never a fake 0


def test_snake_apple_intervals_count_from_the_start_but_not_after_the_last_apple() -> None:
    game = snake_game("m", "m", "a", "m", "a", "m", "m", "m", "hit_wall")
    assert apple_intervals(game) == [3, 2]  # the four steps after the last apple are no interval
    assert apple_intervals(snake_game("m", "hit_self")) == []
    assert apple_intervals(snake_game("a", "m", "w")) == [1, 2]  # the winning step eats the last apple


def test_snake_statistics() -> None:
    won_fast = snake_game("a", "m", "w")  # 3 steps
    won_slow = snake_game("m", "a", "m", "m", "m", "w")  # 6 steps
    died = snake_game("m", "m", "m", "a", "m", "starved")
    stats = snake_statistics([won_fast, won_slow, died])
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["mean_steps_to_win"] == 4.5 and stats["median_steps_to_win"] == 4.5
    intervals = [1, 2, 2, 4, 4]
    assert stats["mean_steps_between_apples"] == pytest.approx(sum(intervals) / 5)
    assert stats["median_steps_between_apples"] == 2


def test_snake_statistics_are_not_available_without_wins_or_apples() -> None:
    stats = snake_statistics([snake_game("m", "hit_wall"), snake_game("m", "m", "hit_self")])
    assert stats == {
        "win_rate": 0.0, "mean_steps_to_win": None, "median_steps_to_win": None,
        "mean_steps_between_apples": None, "median_steps_between_apples": None,
    }


def test_lander_statistics() -> None:
    records = [
        record("landed", ["flying"] * 9 + ["landed"]),
        record("landed", ["flying"] * 19 + ["landed"]),
        record("crashed", ["flying", "crashed"]),
        record("timeout", ["flying"] * 3 + ["timeout"]),
        record(STEP_LIMIT, ["flying"] * 5),
    ]
    stats = lander_statistics(records)
    assert stats == {"landing_rate": 0.4, "crash_rate": 0.2, "flew_off_rate": 0.0, "timeout_rate": 0.2, "mean_steps_to_land": 15.0}
    assert lander_statistics([record("crashed", ["crashed"])])["mean_steps_to_land"] is None


def test_evaluation_statistics_put_the_game_metrics_first_and_check_their_keys() -> None:
    snake = get_game("snake")
    stats = evaluation_statistics(snake, [snake_game("a", "hit_wall")])
    assert list(stats) == [definition.key for definition in metric_definitions(snake)]
    assert list(stats)[:2] == ["win_rate", "mean_steps_to_win"] and stats["mean_score"] == 1.0
    assert all(value is None for key, value in evaluation_statistics(snake, []).items() if key != "games")
    wrong = dataclasses.replace(snake, episode_metrics=GameMetrics(snake.episode_metrics.definitions, lambda records: {"win_rate": 1.0}))
    with pytest.raises(ValueError, match="defines"):
        evaluation_statistics(wrong, [snake_game("hit_wall")])
    assert [d.key for d in metric_definitions(get_game("lunar_lander"))][0] == "landing_rate"


def test_metric_definitions_are_validated() -> None:
    definition = MetricDefinition("x", "x", "an x")
    assert metric_problems(GameMetrics((definition,), lambda records: {})) == []
    problems = metric_problems(GameMetrics((definition, definition, MetricDefinition("mean_score", "m", "d"), MetricDefinition("Bad key", "", "")), None))
    assert any("callable" in p for p in problems) and any("distinct" in p for p in problems)
    assert any("generic" in p for p in problems) and any("a-z" in p for p in problems) and any("label" in p for p in problems)


def test_format_value_and_table() -> None:
    assert format_value(MetricDefinition("r", "r", "d", format="rate"), 0.125) == "12.5 %"
    assert format_value(MetricDefinition("s", "s", "d", unit="steps"), None) == "N/A"
    assert format_value(MetricDefinition("g", "g", "d", format="integer"), 1200.0) == "1,200"
    evaluation = Evaluation("manual", 10, "checkpoint_0000000010.pt", EvalSettings(2, 0, 50), {"games": 2.0, "mean_score": None}, {"won": 2}, 0.5)
    table = evaluation_table(evaluation, GENERIC_METRICS[:2])
    assert "mean score" in table and "N/A" in table and "won 2" in table and "checkpoint_0000000010.pt" in table


# --------------------------------------------------------------------------- schedule, records, reproduction


@pytest.mark.parametrize(
    ("previous", "steps", "interval", "is_last", "due"),
    [(0, 64, 100, False, False), (64, 128, 100, False, True), (128, 192, 100, False, False), (0, 300, 100, False, True),
     (0, 64, 100, True, True), (0, 640, 0, True, False), (100, 200, 100, False, True)],
)
def test_evaluation_due(previous, steps, interval, is_last, due) -> None:
    assert evaluation_due(previous, steps, interval, is_last) is due


def test_eval_config_validation_and_settings() -> None:
    with pytest.raises(ValueError):
        EvalConfig(interval=-1)
    with pytest.raises(ValueError):
        EvalConfig(games=0)
    assert not EvalConfig().enabled and EvalConfig(interval=5).enabled
    settings = settings_from_config(EvalConfig(games=7, seed=3), "cpu", games=None, seed=9)
    assert (settings.games, settings.seed, settings.device) == (7, 9, "cpu")
    assert settings.to_json()["policy"] == "greedy" and EvalSettings.from_json(settings.to_json()) == settings


def test_records_round_trip_and_bad_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "evaluations.jsonl"
    evaluation = Evaluation("training", 64, None, EvalSettings(3, 1, 9), {"games": 3.0, "win_rate": None}, {"won": 3}, 0.1)
    append_evaluation(path, evaluation)
    with path.open("a") as file:
        file.write('{"truncated": \n{"source": "training"}\n')
    append_evaluation(path, dataclasses.replace(evaluation, source="manual"))
    loaded = read_evaluations(path)
    assert loaded == [evaluation, dataclasses.replace(evaluation, source="manual")]
    assert json.loads(path.read_text().splitlines()[0])["statistics"]["win_rate"] is None
    assert read_evaluations(tmp_path / "missing.jsonl") == []
    with pytest.raises(ValueError):
        Evaluation("guess", 0, None, EvalSettings(1, 1, 1), {}, {}, 0.0)


def test_find_reproduction() -> None:
    old = Evaluation("training", 64, "c.pt", EvalSettings(3, 1, 9), {"mean_score": 1.0}, {"won": 3}, 0.1)
    same = dataclasses.replace(old, source="manual", seconds=0.3)
    assert find_reproduction([old], same).identical
    other_numbers = dataclasses.replace(same, statistics={"mean_score": 2.0})
    assert not find_reproduction([old], other_numbers).identical
    gpu = dataclasses.replace(same, settings=dataclasses.replace(old.settings, device="cuda"))
    assert find_reproduction([old], gpu).same_device is False
    assert find_reproduction([old], dataclasses.replace(same, checkpoint="d.pt")) is None
    assert find_reproduction([old], dataclasses.replace(same, settings=EvalSettings(4, 1, 9))) is None


# --------------------------------------------------------------------------- playing


class CountingEnv:
    """A ``SingleEnv`` whose games last ``seed % 5 + 1`` steps, or forever with ``endless``; the score is the seed."""

    num_actions = 2
    observation_shape = (1,)
    history_length = frame_channels = 1
    action_channels = 0

    def __init__(self, endless: bool = False) -> None:
        self.endless = endless
        self.seeds: list[int] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seeds.append(seed)
        self.left = seed % 5 + 1
        return np.zeros(1, dtype=np.float32)

    def step(self, action: int) -> EnvStep:
        self.left -= 1
        done = self.left == 0 and not self.endless
        return EnvStep(np.zeros(1, dtype=np.float32), 1.0, done, False, "end" if done else "tick", float(self.seeds[-1]))

    def view(self) -> dict:
        return {}


class FirstActionPolicy(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> PolicyOutput:
        batch = observations.shape[0]
        return PolicyOutput(torch.tensor([[1.0, 0.0]]).repeat(batch, 1), torch.zeros(batch))


def test_play_games_records_every_game_in_order_and_caps_endless_ones() -> None:
    envs: list[CountingEnv] = []

    def factory() -> CountingEnv:
        envs.append(CountingEnv())
        return envs[-1]

    policy = FirstActionPolicy().train()
    progress = []
    records = play_games(policy, factory, EvalSettings(games=7, seed=5, max_episode_steps=100, parallel_games=3), lambda d, t: progress.append((d, t)))
    assert policy.training  # mode restored
    assert len(envs) == 3 and progress[-1] == (7, 7) and len(progress) == 7
    assert [r.score for r in records] == [float(env_seed(5, k)) for k in range(7)]  # game k plays seed k, in order
    assert all(r.length == env_seed(5, k) % 5 + 1 and r.outcome == "end" for k, r in enumerate(records))
    capped = play_games(policy, lambda: CountingEnv(endless=True), EvalSettings(games=2, seed=0, max_episode_steps=12))
    assert [(r.outcome, r.length, len(r.events)) for r in capped] == [(STEP_LIMIT, 12, 12)] * 2


TINY_ALGORITHMS = {
    "ppo": {"rollout_length": 16, "num_epochs": 1, "num_minibatches": 2},
    "dqn": {"buffer_size": 200, "batch_size": 8, "learning_starts": 16, "target_update_interval": 32},
}


def tiny_snake_config(algorithm: str = "ppo", **evaluation):
    return experiment_from_tables({
        "game": {"width": 4, "height": 4, "initial_length": 2, "max_steps_without_food": 16},
        "model": {"kind": "cnn", "conv_channels": 8, "conv_layers": 1},
        "train": {"algorithm": algorithm, "total_env_steps": 192, "device": "cpu", "log_interval": 100_000, "checkpoint_interval": 100_000},
        algorithm: TINY_ALGORITHMS[algorithm],
        "env": {"num_envs": 4},
        "evaluation": evaluation,
    })


def test_greedy_evaluation_of_a_policy_is_deterministic() -> None:
    config = tiny_snake_config()
    torch.manual_seed(0)
    policy = make_policy(config, make_env(config))
    settings = EvalSettings(games=10, seed=3, max_episode_steps=200, parallel_games=4)
    first = play_games(policy, lambda: make_env(config), settings)
    torch.manual_seed(123)  # global streams do not matter: nothing is sampled
    assert play_games(policy, lambda: make_env(config), settings) == first


def train_tiny(config, checkpoint_dir: Path | None, hooks=()) -> Trainer:
    config = dataclasses.replace(config, train=dataclasses.replace(config.train, checkpoint_dir=checkpoint_dir))
    env = make_vec_env(config)
    torch.manual_seed(config.train.seed)
    policy = make_policy(config, env)
    trainer = make_trainer(config, policy, env, NullLogger(), hooks)
    trainer.train()
    return trainer


def test_evaluating_during_training_changes_nothing_about_training(tmp_path: Path) -> None:
    config = tiny_snake_config(interval=100, games=6)
    plain = train_tiny(config, None)
    log = tmp_path / "evaluations.jsonl"
    hook = PeriodicEvaluation(get_game("snake"), lambda: make_env(config), config.evaluation, log, announce=lambda line: None)
    evaluated = train_tiny(config, tmp_path, hooks=[hook])
    for a, b in zip(plain.policy.state_dict().values(), evaluated.policy.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(plain.learner.optimizer.state_dict()["state"].values(), evaluated.learner.optimizer.state_dict()["state"].values(), strict=True):
        torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
    # 64 steps per update, 3 updates: crossing 100 at 128, then the end at 192.
    evaluations = read_evaluations(log)
    assert [(e.env_steps, e.checkpoint, e.source) for e in evaluations] == [
        (128, "checkpoint_0000000128.pt", "training"),
        (192, "checkpoint_0000000192.pt", "training"),
    ]
    assert (tmp_path / "checkpoint_0000000128.pt").is_file()  # saved for the evaluation (interval is 100 updates)
    assert evaluations[0].statistics["games"] == 6 and sum(evaluations[0].outcomes.values()) == 6
    assert evaluated.policy.training  # mode restored after evaluating


def test_evaluating_during_dqn_training_changes_nothing_about_training(tmp_path: Path) -> None:
    """Exploration and replay sampling use their own seeded generators; evaluation touches neither."""
    config = tiny_snake_config("dqn", interval=100, games=6)
    plain = train_tiny(config, None)
    hook = PeriodicEvaluation(get_game("snake"), lambda: make_env(config), config.evaluation, tmp_path / "e.jsonl", announce=lambda line: None)
    evaluated = train_tiny(config, tmp_path, hooks=[hook])
    assert plain.learner.report()["gradient_steps"] > 0
    for a, b in zip(plain.policy.state_dict().values(), evaluated.policy.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(plain.learner.extra_state()["target"].values(), evaluated.learner.extra_state()["target"].values(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # 4 env steps per iteration: crossing 100 at 100, then the end at 192; greedy play on the Q-values.
    assert [e.env_steps for e in read_evaluations(tmp_path / "e.jsonl")] == [100, 192]
