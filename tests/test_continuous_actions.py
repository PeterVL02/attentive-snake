"""Continuous (box) actions: the action-space contract, the Gaussian head, PPO on real-valued actions, what refuses
them (DQN, history_actions, the world-model kinds), Lunar Lander's and Atari's continuous rules, evaluation and watching."""

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Independent, Normal

from attentive_snake.config import EnvConfig, ExperimentConfig, save_experiment
from attentive_snake.config.action_compatibility import continuous_action_problems
from attentive_snake.config.fork_rules import locked_field_changes
from attentive_snake.env.action_space import ContinuousActions, DiscreteActions, action_space_of
from attentive_snake.env.history_env import HistoryEnv
from attentive_snake.evaluation.player import play_games
from attentive_snake.evaluation.settings import EvalSettings
from attentive_snake.factory import make_env, make_policy, make_trainer, make_vec_env
from attentive_snake.games.atari.config import AtariRules
from attentive_snake.games.atari.joystick import CONTINUOUS_ACTIONS as STICK, joystick_press
from attentive_snake.games.atari.reward import AtariReward
from attentive_snake.games.lunar_lander.config import LanderRules
from attentive_snake.games.lunar_lander.game import GAME as LANDER
from attentive_snake.games.lunar_lander.reward import LanderReward
from attentive_snake.games.registry import get_game
from attentive_snake.games.validation import spec_problems
from attentive_snake.models.acting import action_readout, can_sample, greedy_actions, sampled_actions
from attentive_snake.models.action_distribution import policy_distribution
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import KIND_INFO, POLICY_KINDS, build_policy
from attentive_snake.models.types import GaussianPolicyOutput, PolicyOutput
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.run_dir import config_path
from attentive_snake.training import NullLogger, TrainConfig
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.ppo.config import PpoConfig
from attentive_snake.training.ppo.loss import ppo_loss
from attentive_snake.training.ppo.rollout import RolloutBuffer, flatten_rollout
from attentive_snake.viz import diagram
from attentive_snake.watch import WatchSession

CONTINUOUS = LanderRules(continuous=True)
SMALL = dict(d_model=16, num_heads=2, num_layers=1, conv_channels=8, conv_layers=1)
CONTINUOUS_KINDS = [kind for kind in POLICY_KINDS if not KIND_INFO[kind].needs_discrete_actions]


def lander_config(kind: str = "mlp", algorithm: str = "ppo", history_actions: bool = False, rules=CONTINUOUS, **train) -> ExperimentConfig:
    history = 3 if KIND_INFO[kind].uses_history else 1
    settings = PpoConfig(rollout_length=8, num_minibatches=2) if algorithm == "ppo" else DqnConfig()
    return ExperimentConfig(
        game_name="lunar_lander",
        game=rules,
        reward=LanderReward(),
        model=ModelConfig(kind=kind, **SMALL),
        train=TrainConfig(algorithm=algorithm, total_env_steps=32, device="cpu", **train),
        algorithm_settings=settings,
        env=EnvConfig(num_envs=2, history_length=history, history_actions=history_actions),
    )


# --------------------------------------------------------------------------- the contract


def test_action_spaces_say_their_shape_and_turn_policy_output_into_game_actions():
    discrete = DiscreteActions(4)
    assert (discrete.continuous, discrete.shape, discrete.env_action(np.int64(2))) == (False, (), 2)
    box = ContinuousActions(low=(-1.0, 0.0), high=(1.0, 2.0))
    assert (box.continuous, box.shape, box.dimension) == (True, (2,), 2)
    played = box.env_action(torch.tensor([3.0, -0.5]).numpy())
    assert played.dtype == np.float32 and played.tolist() == [1.0, 0.0]  # clipped to the bounds
    for low, high in [((), ()), ((0.0,), (0.0,)), ((0.0, 1.0), (1.0,))]:
        with pytest.raises(ValueError):
            ContinuousActions(low, high)


def test_an_env_without_an_action_space_is_discrete_and_the_wrappers_pass_it_on():
    assert action_space_of(SimpleNamespace(num_actions=3)) == DiscreteActions(3)
    discrete = make_vec_env(dataclasses.replace(lander_config(), game=LanderRules()))
    assert discrete.action_space == DiscreteActions(4) and discrete.envs[0].action_space == DiscreteActions(4)
    vec = make_vec_env(lander_config())
    assert vec.action_space == LANDER.make_env(CONTINUOUS, LanderReward(), 0).action_space and vec.num_actions == 2


def test_lunar_lander_continuous_rule_gives_two_named_throttles_and_clips_them():
    env = LANDER.make_env(CONTINUOUS, LanderReward(), 0)
    assert env.action_space == ContinuousActions((-1.0, -1.0), (1.0, 1.0)) and env.num_actions == 2
    assert env.action_labels == ("main engine", "side engines")
    assert LANDER.has_continuous_actions(CONTINUOUS) and not LANDER.has_continuous_actions(LanderRules())
    assert not get_game("snake").has_continuous_actions(get_game("snake").rules_type())
    single = HistoryEnv(env)
    single.reset(0)
    step = single.step(np.array([5.0, -5.0]))  # far beyond the bounds: the game gets [1, -1]
    assert step.observation.shape == (8,) and np.isfinite(step.reward)
    assert spec_problems(LANDER, "lunar_lander") == []


def test_history_actions_need_discrete_actions():
    with pytest.raises(ValueError, match="history_actions needs discrete actions"):
        HistoryEnv(LANDER.make_env(CONTINUOUS, LanderReward(), 0), history_length=2, history_actions=True)


def test_continuous_is_locked_on_fork():
    assert locked_field_changes(dataclasses.replace(lander_config(), game=LanderRules()), lander_config()) == ["game.continuous"]
    atari = ExperimentConfig(game_name="atari", game=AtariRules(), reward=AtariReward(), model=ModelConfig(kind="cnn"))
    assert locked_field_changes(atari, dataclasses.replace(atari, game=AtariRules(continuous=True))) == ["game.continuous"]


# --------------------------------------------------------------------------- what refuses continuous actions


def test_dqn_history_actions_and_world_models_are_refused_with_plain_reasons():
    with pytest.raises(ValueError, match="DQN learns one value per action"):
        lander_config(algorithm="dqn")
    with pytest.raises(ValueError, match="env.history_actions"):
        lander_config(kind="temporal_transformer", history_actions=True)
    for kind in POLICY_KINDS:
        if KIND_INFO[kind].needs_discrete_actions:
            with pytest.raises(ValueError, match=f"the {kind} model predicts"):
                lander_config(kind=kind)
    assert {kind for kind in POLICY_KINDS if KIND_INFO[kind].needs_discrete_actions} == {"jepa", "jepa_attention", "jepa_tokens", "jepa_temporal"}
    problems = continuous_action_problems("lunar_lander", CONTINUOUS, "dqn", "jepa", True)
    assert len(problems) == 3
    assert continuous_action_problems("lunar_lander", LanderRules(), "dqn", "jepa", True) == []  # discrete: nothing to refuse
    lander_config(algorithm="dqn", rules=LanderRules())


def test_the_algorithm_picks_the_head_by_the_action_space():
    from attentive_snake.training.algorithms import get_algorithm

    ppo, dqn = get_algorithm("ppo"), get_algorithm("dqn")
    assert (ppo.head_for(PpoConfig(), False), ppo.head_for(PpoConfig(), True)) == ("actor_critic", "gaussian")
    assert ppo.trains_continuous_actions and not dqn.trains_continuous_actions
    with pytest.raises(ValueError, match="discrete actions only"):
        dqn.head_for(DqnConfig(), True)


# --------------------------------------------------------------------------- the Gaussian head


SPECS = {"grid": ObservationSpec("grid", (6, 5, 6), 2, 0), "vector": ObservationSpec("vector", (10,), 2, 0)}


@pytest.mark.parametrize("observation", sorted(SPECS))
@pytest.mark.parametrize("kind", CONTINUOUS_KINDS)
def test_every_kind_body_works_with_the_gaussian_head(kind, observation):
    torch.manual_seed(0)
    spec = SPECS[observation]
    if observation not in KIND_INFO[kind].inputs:
        pytest.skip(f"{kind} does not read {observation} observations")
    policy = build_policy(ModelConfig(kind=kind, **SMALL), spec, 3, "gaussian")
    plain = build_policy(ModelConfig(kind=kind, **SMALL), spec, 3, "actor_critic")
    out = policy(torch.rand(4, *spec.shape))
    assert isinstance(out, GaussianPolicyOutput)
    assert out.mean.shape == out.log_std.shape == (4, 3) and out.value.shape == (4,)
    assert torch.all(out.log_std == 0.0) and out.mean.abs().max() < 0.5  # std 1, mean near 0 at the start
    head_keys = {"net.action_mean.linear.weight", "net.action_mean.linear.bias", "net.action_log_std.log_std"}
    trunk = {key for key in policy.state_dict() if key not in head_keys}
    assert trunk == {key for key in plain.state_dict() if not key.startswith("net.policy_head.")}  # the same body
    (out.mean.sum() + out.log_std.sum() + out.value.sum()).backward()
    assert all(p.grad is not None for p in policy.parameters())
    assert "LogStd" in diagram(policy)


def test_the_distribution_sums_log_probabilities_and_entropy_over_the_dimensions():
    mean, log_std = torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, math.log(2.0)]])
    distribution = policy_distribution(GaussianPolicyOutput(mean, log_std, torch.zeros(1)))
    action = torch.tensor([[0.5, 0.0]])
    expected = Normal(0.0, 1.0).log_prob(torch.tensor(0.5)) + Normal(1.0, 2.0).log_prob(torch.tensor(0.0))
    torch.testing.assert_close(distribution.log_prob(action), expected.reshape(1))
    assert distribution.batch_shape == (1,) and distribution.event_shape == (2,)
    torch.testing.assert_close(distribution.entropy(), (Normal(0.0, 1.0).entropy() + Normal(1.0, 2.0).entropy()).reshape(1))
    assert isinstance(policy_distribution(PolicyOutput(torch.zeros(1, 3), torch.zeros(1))), Categorical)


def test_acting_plays_the_mean_greedily_samples_around_it_and_reads_out_mean_and_spread():
    out = GaussianPolicyOutput(torch.tensor([[0.25, -0.5]]), torch.tensor([[0.0, math.log(0.5)]]), torch.tensor([2.0]))
    assert greedy_actions(out).tolist() == [[0.25, -0.5]] and can_sample(out)
    torch.manual_seed(0)
    assert sampled_actions(out).shape == (1, 2) and sampled_actions(out).tolist() != [[0.25, -0.5]]
    readout = action_readout(out).to_dict()
    assert readout == {"kind": "action_vector", "label": "mean action", "values": [0.25, -0.5], "state_value": 2.0, "spread": [1.0, 0.5]}


def test_ppo_loss_on_a_gaussian_raises_the_likelihood_of_good_actions():
    mean = torch.zeros(1, 2, requires_grad=True)
    log_std = torch.zeros(1, 2)
    distribution = Independent(Normal(mean, log_std.exp()), 1)
    action = torch.tensor([[1.0, -1.0]])
    old = distribution.log_prob(action).detach()
    terms = ppo_loss(distribution, torch.zeros(1), action, old, torch.tensor([1.0]), torch.zeros(1), 0.2, 0.0, 0.0)
    terms.total.backward()
    assert mean.grad[0, 0] < 0 < mean.grad[0, 1]  # gradient descent moves the mean towards the action taken


def test_the_rollout_stores_real_valued_action_vectors():
    buffer = RolloutBuffer.allocate(3, 2, (8,), torch.device("cpu"), action_space=ContinuousActions((-1.0,) * 2, (1.0,) * 2))
    assert buffer.actions.shape == (3, 2, 2) and buffer.actions.dtype == torch.float32
    assert RolloutBuffer.allocate(3, 2, (8,), torch.device("cpu"), action_space=DiscreteActions(4)).actions.dtype == torch.int64
    zeros = torch.zeros(3, 2)
    assert flatten_rollout(buffer, zeros, zeros).actions.shape == (6, 2)


# --------------------------------------------------------------------------- training, evaluation and watching


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A tiny continuous Lunar Lander PPO run per kind that can have continuous actions (temporal kinds with 3 frames)."""
    root = tmp_path_factory.mktemp("continuous_runs")
    for kind in CONTINUOUS_KINDS:
        config = lander_config(kind, checkpoint_dir=root / kind)
        save_experiment(config, config_path(root / kind))
        env = make_vec_env(config)
        make_trainer(config, make_policy(config, env), env, NullLogger()).train()
    return root


@pytest.mark.parametrize("kind", CONTINUOUS_KINDS)
def test_every_kind_trains_and_is_watched_with_its_mean_action(runs, kind):
    session = WatchSession(runs / kind, seed=0)
    assert json.loads(json.dumps(session.info)) == session.info
    assert session.info["action_names"] == ["main engine", "side engines"]
    assert session.info["action_bounds"] == {"low": [-1.0, -1.0], "high": [1.0, 1.0]}
    for frame in [session.reset(), *(session.step() for _ in range(3))]:
        readout = frame.to_dict()["action_values"]
        assert readout["kind"] == "action_vector" and len(readout["values"]) == len(readout["spread"]) == 2
    session.set_options(sample=True)
    session.step()


def test_a_trained_continuous_policy_moved_its_parameters_and_evaluates_deterministically(runs):
    config = lander_config("mlp")
    policy = make_policy(config, make_env(config))
    settings = EvalSettings(games=2, max_episode_steps=50, parallel_games=2, seed=3, device="cpu")
    first = play_games(policy, lambda: make_env(config), settings)
    assert first == play_games(policy, lambda: make_env(config), settings) and len(first) == 2
    assert WatchSession(runs / "mlp", seed=0).info["env_steps"] == 32


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def log(self, env_steps: int, metrics) -> None:
        self.records.append(dict(metrics))


def test_the_ppo_metrics_of_a_continuous_run_are_finite_and_the_spread_is_learned(tmp_path):
    config = lander_config("mlp", checkpoint_dir=tmp_path, log_interval=1)
    env = make_vec_env(config)
    policy = make_policy(config, env)
    logger = RecordingLogger()
    make_trainer(config, policy, env, logger).train()
    ppo = [record for record in logger.records if "entropy" in record]
    assert ppo and all(math.isfinite(record[key]) for record in ppo for key in ("policy_loss", "value_loss", "entropy", "approx_kl"))
    assert ppo[0]["entropy"] == pytest.approx(2 * Normal(0.0, 1.0).entropy().item(), abs=0.05)  # two numbers, std 1 at the start
    assert not torch.all(policy.net["action_log_std"].log_std == 0.0)


# --------------------------------------------------------------------------- Atari


@pytest.mark.parametrize(
    "action, name, strength",
    [
        ((0.0, 0.0, 0.0), "NOOP", 0.0),
        ((0.0, 0.0, 1.0), "FIRE", 0.0),
        ((1.0, 0.0, 0.0), "RIGHT", 1.0),
        ((1.0, math.pi, 0.0), "LEFT", 1.0),
        ((1.0, math.pi / 2, 1.0), "UPFIRE", 1.0),
        ((1.0, -math.pi / 4, 0.0), "DOWNRIGHT", 1.0),
        ((0.4, 0.0, 0.0), "NOOP", 0.4),  # pushed less than half way: no direction
    ],
)
def test_the_continuous_joystick_reads_push_direction_and_fire_as_ale_does(action, name, strength):
    pressed, force = joystick_press(np.array(action, dtype=np.float32))
    assert pressed == name and force == pytest.approx(strength)


def test_atari_continuous_rule_plays_the_stick_with_its_strength():
    game = get_game("atari")
    env = game.make_env(AtariRules(continuous=True, sticky_actions=0.0, screen_size=36), AtariReward(), 0)
    assert env.action_space == STICK and env.num_actions == 3 and env.action_labels == ("stick push", "stick direction", "fire")
    assert game.has_continuous_actions(AtariRules(continuous=True))
    real, pressed = env._ale, []
    env._ale = SimpleNamespace(**{name: getattr(real, name) for name in ("game_over", "lives", "getScreenGrayscale")})
    env._ale.act = lambda action, strength=1.0: pressed.append((action.name, round(float(strength), 3))) or real.act(action, strength)
    single = HistoryEnv(env)
    single.reset(0)
    single.step(np.array([2.0, 0.0, 1.0]))  # push clipped to 1: right + fire at full strength
    assert set(pressed) == {("RIGHTFIRE", 1.0)} and len(pressed) == 4


def test_atari_continuous_trains_a_few_steps_with_ppo(tmp_path):
    config = ExperimentConfig(
        game_name="atari",
        game=AtariRules(continuous=True, screen_size=36, max_episode_steps=20),
        reward=AtariReward(),
        model=ModelConfig(kind="cnn", d_model=16, conv_channels=8),
        train=TrainConfig(total_env_steps=16, device="cpu", checkpoint_dir=tmp_path),
        algorithm_settings=PpoConfig(rollout_length=8, num_minibatches=2, num_epochs=1),
        env=EnvConfig(num_envs=2),
    )
    env = make_vec_env(config)
    policy = make_policy(config, env)
    assert policy.head_name == "gaussian"
    make_trainer(config, policy, env, NullLogger()).train()
