"""PPO's observation normalisation, return normalisation and clipped value loss."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.fork_rules import locked_fields, require_forkable
from attentive_snake.evaluation.checkpoint_evaluation import evaluate_checkpoint
from attentive_snake.evaluation.settings import EvalSettings
from attentive_snake.factory import make_env, make_policy, make_trainer, make_vec_env
from attentive_snake.config import save_experiment
from attentive_snake.models.auxiliary import TransitionBatch, TransitionSequences
from attentive_snake.models.observation_normalizer import CLIP, ObservationNormalizer
from attentive_snake.models.running_statistics import RunningStatistics
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.run_dir import config_path
from attentive_snake.training.checkpoint import latest_checkpoint, read_checkpoint
from attentive_snake.training.fork import fork_from_checkpoint
from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.training.ppo.loss import ppo_loss
from attentive_snake.training.ppo.normalized_transitions import normalized_transitions
from attentive_snake.training.ppo.return_normalizer import ReturnNormalizer
from attentive_snake.training.ppo.value_loss import clipped_value_loss
from attentive_snake.training.resume import resume_from_checkpoint
from attentive_snake.watch.session import WatchSession

STATISTICS = ("observation_normalizer.statistics.mean", "observation_normalizer.statistics.variance", "observation_normalizer.statistics.count")


def lander_tables(run_dir: Path, kind: str = "mlp", **ppo) -> dict:
    return {
        "game": {"name": "lunar_lander"},
        "model": {"kind": kind, "d_model": 16, "num_heads": 2, "num_layers": 1, "mlp_ratio": 2, "conv_channels": 8, "conv_layers": 1},
        "train": {"total_env_steps": 64, "log_interval": 1000, "checkpoint_interval": 32, "device": "cpu", "checkpoint_dir": str(run_dir), "bootstrap_truncated": True},
        "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2, **ppo},
        "env": {"num_envs": 2},
    }


def trained(run_dir: Path, **ppo):
    """A short finished run in ``run_dir``: its config and trainer."""
    config = experiment_from_tables(lander_tables(run_dir, **ppo))
    save_experiment(config, config_path(run_dir))
    trainer = trainer_of(config)
    trainer.train()
    return config, trainer


def trainer_of(config):
    env = make_vec_env(config)
    return make_trainer(config, make_policy(config, env), env, logger=_Silent())


class _Silent:
    def log(self, env_steps: int, record: dict) -> None:
        pass


# ---------------------------------------------------------------- running statistics


def test_running_statistics_equal_those_of_all_the_data_at_once():
    data = torch.randn(600, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * torch.tensor([1.0, 5.0, 0.1], dtype=torch.float64) + 7.0
    statistics = RunningStatistics((3,))
    for batch in data.split(37):
        statistics.update(batch)
    assert statistics.mean.numpy() == pytest.approx(data.mean(dim=0).numpy(), abs=1e-4)
    assert statistics.variance.numpy() == pytest.approx(data.var(dim=0, correction=0).numpy(), rel=1e-3)
    assert float(statistics.count) == pytest.approx(600.0, abs=1e-3)


def test_running_statistics_of_a_scalar_stream_and_their_state_round_trip():
    statistics = RunningStatistics()
    statistics.update(torch.tensor([1.0, 3.0]))
    copy = RunningStatistics()
    copy.load_state_dict(statistics.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(statistics.state_dict().values(), copy.state_dict().values()))
    assert statistics.mean.dtype == torch.float64 and float(statistics.mean) == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------- observations


def spec(history: int = 1, actions: int = 0, numbers: int = 2) -> ObservationSpec:
    return ObservationSpec("vector", (history * (numbers + actions),), history, actions)


def test_observations_become_zero_mean_unit_spread_and_are_clipped():
    normalizer = ObservationNormalizer(spec())
    data = torch.randn(4000, 2, generator=torch.Generator().manual_seed(1)) * torch.tensor([100.0, 0.01]) + torch.tensor([50.0, -3.0])
    normalizer.update(data)
    scaled = normalizer(data)
    assert scaled.mean(dim=0).numpy() == pytest.approx([0.0, 0.0], abs=1e-2)
    assert scaled.std(dim=0).numpy() == pytest.approx([1.0, 1.0], abs=1e-2)
    assert normalizer(torch.tensor([[1e6, -1e6]])).tolist() == [[CLIP, -CLIP]]


def test_action_channels_pass_unchanged_and_all_frames_share_one_scale():
    normalizer = ObservationNormalizer(spec(history=2, actions=2))
    normalizer.statistics.mean.copy_(torch.tensor([1.0, 2.0], dtype=torch.float64))
    normalizer.statistics.variance.copy_(torch.tensor([4.0, 4.0], dtype=torch.float64))
    observation = torch.tensor([[3.0, 2.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0]])
    assert normalizer(observation).numpy() == pytest.approx(np.array([[1.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.0, 1.0]]), abs=1e-6)


def test_statistics_learn_from_the_newest_frame_without_its_action_channels():
    normalizer = ObservationNormalizer(spec(history=2, actions=1))
    padded_start = torch.tensor([[0.0, 0.0, 0.0, 4.0, 6.0, 1.0], [0.0, 0.0, 0.0, 8.0, 10.0, 1.0]])
    normalizer.update(padded_start)
    assert normalizer.statistics.mean.numpy() == pytest.approx([6.0, 8.0], abs=1e-3)


@pytest.mark.parametrize("game", ["snake", "atari", "minatar"])
def test_games_that_are_not_described_by_numbers_refuse_observation_normalisation(game: str):
    with pytest.raises(ValueError, match="normalize_observations only works"):
        experiment_from_tables({"game": {"name": game}, "ppo": {"normalize_observations": True}})


def test_dqn_settings_have_no_normalisation_and_the_new_settings_start_off():
    assert not {"normalize_observations", "normalize_returns", "clip_value_loss"} & {f.name for f in dataclasses.fields(experiment_from_tables({"train": {"algorithm": "dqn"}}).algorithm_settings)}
    defaults = PpoConfig()
    assert (defaults.normalize_observations, defaults.normalize_returns, defaults.clip_value_loss) == (False, False, False)


def test_off_adds_no_state_dict_keys_and_on_adds_only_the_statistics(tmp_path: Path):
    off = make_policy(experiment_from_tables(lander_tables(tmp_path)), make_env(experiment_from_tables(lander_tables(tmp_path))))
    config = experiment_from_tables(lander_tables(tmp_path, normalize_observations=True))
    on = make_policy(config, make_env(config))
    assert off.observation_normalizer is None
    assert set(on.state_dict()) - set(off.state_dict()) == set(STATISTICS) and set(off.state_dict()) < set(on.state_dict())


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_every_way_into_the_network_reads_normalised_observations(tmp_path: Path, kind: str):
    config = experiment_from_tables(lander_tables(tmp_path, kind=kind, normalize_observations=True))
    policy = make_policy(config, make_env(config)).eval()
    observations = torch.randn(4, 8, generator=torch.Generator().manual_seed(2)) * 30.0
    policy.observation_normalizer.update(observations)
    seen = []
    first = next(iter(policy.net.values()))
    handle = first.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    policy(observations)
    if kind == "transformer":
        policy.attention_maps(observations)
    handle.remove()
    assert len(seen) == (2 if kind == "transformer" else 1)
    assert all(torch.equal(batch, policy.observation_normalizer(observations)) for batch in seen)


def test_an_auxiliary_objectives_transitions_are_normalised_like_the_policys_input():
    normalizer = ObservationNormalizer(spec())
    normalizer.update(torch.tensor([[2.0, 4.0], [4.0, 8.0]]))
    observations, following = torch.tensor([[3.0, 6.0]]), torch.tensor([[4.0, 8.0]])
    later = following.unsqueeze(1)
    sequences = TransitionSequences(torch.zeros((1, 1), dtype=torch.int64), later, torch.ones(1, 1), torch.zeros(1, 1))
    batch = normalized_transitions(normalizer, TransitionBatch(observations, torch.zeros(1, dtype=torch.int64), following, torch.ones(1), sequences))
    assert torch.equal(batch.observations, normalizer(observations)) and torch.equal(batch.next_observations, normalizer(following))
    assert torch.equal(batch.sequences.next_observations, normalizer(later))


# ---------------------------------------------------------------- returns


def test_rewards_are_divided_by_the_spread_of_the_discounted_return_without_subtracting_a_mean():
    normalizer = ReturnNormalizer(2, gamma=0.5, device=torch.device("cpu"))
    returns, seen = np.zeros(2), []
    for reward in ([1.0, 3.0], [2.0, -1.0], [4.0, 5.0]):
        scaled = normalizer(torch.tensor(reward), torch.zeros(2))
        returns = 0.5 * returns + np.array(reward)
        seen.extend(returns)
        expected = np.array(reward) / np.sqrt(_tracked_variance(seen) + 1e-8)
        assert scaled.numpy() == pytest.approx(expected, rel=1e-3)
        assert np.all(np.sign(scaled.numpy()) == np.sign(reward))


def _tracked_variance(values: list[float]) -> float:
    return float(np.var(values))


def test_the_discounted_return_restarts_at_an_episode_end_only_in_that_slot():
    normalizer = ReturnNormalizer(2, gamma=1.0, device=torch.device("cpu"))
    normalizer(torch.tensor([5.0, 5.0]), torch.tensor([1.0, 0.0]))
    assert normalizer._returns.tolist() == [0.0, 5.0]


def test_scaled_rewards_are_clipped_and_keep_their_type():
    normalizer = ReturnNormalizer(1, gamma=0.0, device=torch.device("cpu"))
    for _ in range(50):
        normalizer(torch.tensor([0.001]), torch.zeros(1))
    scaled = normalizer(torch.tensor([1e5]), torch.zeros(1))
    assert scaled.dtype == torch.float32 and abs(float(scaled)) <= 10.0


def test_scores_stay_in_game_units_and_the_statistics_are_in_the_checkpoint(tmp_path: Path):
    _, trainer = trained(tmp_path, normalize_returns=True)
    state = read_checkpoint(latest_checkpoint(tmp_path))
    assert float(state["learner"]["return_statistics"]["count"]) > 1.0
    assert not any(key.startswith("observation_normalizer") for key in state["policy"])
    assert trainer.learner.report()["value_loss"] >= 0.0


def test_lander_episodes_report_raw_rewards_with_return_normalisation(tmp_path: Path):
    """Scaling is the collector's business: the score of a whole game is still in the game's own points."""
    config = experiment_from_tables(lander_tables(tmp_path, normalize_returns=True, rollout_length=120))
    trainer = trainer_of(config)
    finished = trainer.learner.iterate(0)
    assert finished and max(abs(episode.score) for episode in finished) > 20.0


# ---------------------------------------------------------------- value loss


def test_the_clipped_value_loss_takes_the_larger_error():
    values, old, returns = torch.tensor([2.0, 1.1]), torch.tensor([1.0, 1.0]), torch.tensor([3.0, 0.0])
    # First: the prediction moved 1.0, clipped to 1.2: errors 1.0 and 1.8. Second: moved 0.1 (within range): errors 1.1 and 1.1.
    expected = 0.5 * (1.8**2 + 1.1**2) / 2.0
    assert float(clipped_value_loss(values, old, returns, 0.2)) == pytest.approx(expected)


def test_within_the_clip_range_the_clipped_loss_is_the_plain_loss():
    values, returns = torch.tensor([1.0, 2.0]), torch.tensor([0.5, 3.0])
    assert float(clipped_value_loss(values, values.clone(), returns, 0.2)) == pytest.approx(float(0.5 * (values - returns).pow(2).mean()))


def test_ppo_loss_clips_the_value_loss_only_when_given_the_old_values():
    distribution = torch.distributions.Categorical(logits=torch.zeros(2, 2))
    arguments = dict(
        distribution=distribution, values=torch.tensor([2.0, 2.0]), actions=torch.zeros(2, dtype=torch.int64), old_log_probs=distribution.log_prob(torch.zeros(2, dtype=torch.int64)),
        advantages=torch.ones(2), returns=torch.tensor([3.0, 3.0]), clip_coef=0.2, value_coef=0.5, entropy_coef=0.0,
    )
    assert float(ppo_loss(**arguments).value_loss) == pytest.approx(0.5)
    assert float(ppo_loss(**arguments, old_values=torch.tensor([1.0, 1.0])).value_loss) == pytest.approx(0.5 * 1.8**2)


# ---------------------------------------------------------------- checkpoints, resume, fork, evaluation, watching


ALL_ON = {"normalize_observations": True, "normalize_returns": True, "clip_value_loss": True}


def test_resume_restores_every_statistic_exactly(tmp_path: Path):
    config, trainer = trained(tmp_path, **ALL_ON)
    saved = read_checkpoint(latest_checkpoint(tmp_path))
    resumed = trainer_of(config)
    resume_from_checkpoint(resumed, latest_checkpoint(tmp_path))
    for key in STATISTICS:
        assert torch.equal(resumed.policy.state_dict()[key], saved["policy"][key]) and torch.equal(trainer.policy.state_dict()[key], saved["policy"][key])
    assert all(torch.equal(resumed.learner.extra_state()["return_statistics"][key], value) for key, value in saved["learner"]["return_statistics"].items())


def test_a_fork_carries_the_statistics_on_and_must_keep_both_normalisations(tmp_path: Path):
    parent, _ = trained(tmp_path / "parent", **ALL_ON)
    child = trainer_of(parent)
    fork_from_checkpoint(child, latest_checkpoint(tmp_path / "parent"))
    saved = read_checkpoint(latest_checkpoint(tmp_path / "parent"))
    assert child.env_steps == 0 and torch.equal(child.policy.state_dict()[STATISTICS[0]], saved["policy"][STATISTICS[0]])
    assert {"ppo.normalize_observations", "ppo.normalize_returns"} <= set(locked_fields("lunar_lander")) and "ppo.clip_value_loss" not in locked_fields("lunar_lander")
    for name in ("normalize_observations", "normalize_returns"):
        changed = experiment_from_tables(lander_tables(tmp_path, **(ALL_ON | {name: False})))
        with pytest.raises(ValueError, match=f"ppo.{name}"):
            require_forkable(parent, changed)
    require_forkable(parent, experiment_from_tables(lander_tables(tmp_path, **(ALL_ON | {"clip_value_loss": False}))))


def test_evaluation_is_deterministic_uses_the_saved_statistics_and_leaves_them_alone(tmp_path: Path):
    _, trainer = trained(tmp_path, **ALL_ON)
    checkpoint = latest_checkpoint(tmp_path)
    settings = EvalSettings(games=2, seed=3, parallel_games=2, max_episode_steps=60, device="cpu")
    before = {key: value.clone() for key, value in trainer.policy.state_dict().items() if key in STATISTICS}
    first, second = evaluate_checkpoint(tmp_path, checkpoint, settings), evaluate_checkpoint(tmp_path, checkpoint, settings)
    assert first.statistics == second.statistics
    assert all(torch.equal(trainer.policy.state_dict()[key], value) for key, value in before.items())
    assert torch.equal(read_checkpoint(checkpoint)["policy"][STATISTICS[0]], before[STATISTICS[0]])


def test_watching_loads_the_statistics_of_the_checkpoint(tmp_path: Path):
    trained(tmp_path, **ALL_ON)
    session = WatchSession(tmp_path, seed=0, attention=False)
    saved = read_checkpoint(latest_checkpoint(tmp_path))["policy"]
    assert torch.equal(session._policy.observation_normalizer.statistics.mean, saved[STATISTICS[0]])
    session.step()


def test_the_statistics_change_only_between_updates_and_final_observations_are_read_alike(tmp_path: Path):
    config = experiment_from_tables(lander_tables(tmp_path, normalize_observations=True))
    trainer = trainer_of(config)
    statistics = trainer.policy.observation_normalizer.statistics
    assert float(statistics.count) < 1.0
    trainer.learner.iterate(0)
    assert float(statistics.count) == pytest.approx(16.0, abs=1e-3)  # one observation per env and step
    final = torch.tensor([[50.0, -50.0, 3.0, 2.0, 1.0, 0.5, 1.0, 0.0]])
    trainer.policy.eval()
    assert torch.equal(trainer.policy(final).value, trainer.policy.net["value_head"](_features(trainer.policy, final)))


def _features(policy, observations: torch.Tensor) -> torch.Tensor:
    pooled, _ = policy._pooled(policy.observation_normalizer(observations), need_weights=False)
    return pooled
