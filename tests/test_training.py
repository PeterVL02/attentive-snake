import math

import numpy as np
import pytest
import torch
from torch import nn

from attentive_snake.env.types import EpisodeSummary, VecEnvStep
from attentive_snake.models.types import PolicyOutput
from attentive_snake.training import (
    NullLogger,
    TrainConfig,
    Trainer,
    fork_from_checkpoint,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    resume_from_checkpoint,
    save_checkpoint,
)
from attentive_snake.training.collector import RolloutCollector
from attentive_snake.training.metrics import EpisodeStatsWindow
from attentive_snake.training.ppo import explained_variance, normalize, ppo_loss
from attentive_snake.training.rollout import RolloutBuffer, compute_gae, flatten_rollout
from attentive_snake.training.schedule import linear_decay

# --------------------------------------------------------------------------- fakes


class MLPPolicy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], num_actions: int, hidden: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(math.prod(observation_shape), hidden), nn.Tanh())
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor) -> PolicyOutput:
        features = self.body(observations)
        return PolicyOutput(self.policy_head(features), self.value_head(features).squeeze(-1))


class CorridorVecEnv:
    """Walk right along a corridor. Obs: one-hot position as (length, 1, 1).

    Action 1 moves right, action 3 moves left, 0 and 2 stay. Reaching the last
    cell gives +1 and terminates; ``max_steps`` truncates (reward 0).
    """

    num_actions = 4

    def __init__(self, num_envs: int, length: int = 5, max_steps: int = 20) -> None:
        self.num_envs = num_envs
        self.length = length
        self.max_steps = max_steps
        self.observation_shape = (length, 1, 1)
        self.positions = np.zeros(num_envs, dtype=np.int64)
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def _observations(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, *self.observation_shape), dtype=np.float32)
        obs[np.arange(self.num_envs), self.positions, 0, 0] = 1.0
        return obs

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.positions[:] = 0
        self.steps[:] = 0
        return self._observations()

    def step(self, actions: np.ndarray) -> VecEnvStep:
        moves = np.where(actions == 1, 1, np.where(actions == 3, -1, 0))
        self.positions = np.clip(self.positions + moves, 0, self.length - 1)
        self.steps += 1
        terminated = self.positions == self.length - 1
        truncated = ~terminated & (self.steps >= self.max_steps)
        rewards = terminated.astype(np.float32)
        finished = []
        ended = np.flatnonzero(terminated | truncated)
        final_observations = {int(i): obs for i, obs in zip(ended, self._observations()[ended], strict=True)}
        for i in ended:
            event = "won" if terminated[i] else "starved"
            finished.append(EpisodeSummary(int(i), int(terminated[i]), int(self.steps[i]), float(rewards[i]), event))
            self.positions[i] = 0
            self.steps[i] = 0
        return VecEnvStep(self._observations(), rewards, terminated, truncated, tuple(finished), final_observations)


class ScriptedVecEnv:
    """Observation = global step counter; reward = 10 * counter + env index; episodes last 3 steps."""

    num_actions = 4
    observation_shape = (1, 1, 1)

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.counter = 0
        self.received_actions: list[np.ndarray] = []

    def _observations(self) -> np.ndarray:
        return np.full((self.num_envs, 1, 1, 1), self.counter, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.counter = 0
        return self._observations()

    def step(self, actions: np.ndarray) -> VecEnvStep:
        self.received_actions.append(actions.copy())
        rewards = (10.0 * self.counter + np.arange(self.num_envs)).astype(np.float32)
        self.counter += 1
        terminated = np.full(self.num_envs, self.counter % 3 == 0)
        truncated = np.zeros(self.num_envs, dtype=bool)
        finals = {i: obs for i, obs in enumerate(self._observations())} if terminated.any() else {}
        return VecEnvStep(self._observations(), rewards, terminated, truncated, (), finals)


# --------------------------------------------------------------------------- GAE & buffer


def test_gae_matches_hand_computation_with_episode_boundary() -> None:
    # env 0: episode ends after step 1; env 1: no boundary.
    rewards = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0]])
    values = torch.tensor([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
    dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    bootstrap = torch.tensor([2.0, 10.0])

    advantages, returns = compute_gae(rewards, values, dones, bootstrap, gamma=0.9, gae_lambda=0.8)

    # env 0, by hand (gamma=0.9, lambda=0.8):
    # t=2: delta = 3 + 0.9*2.0 - 1.5 = 3.3                 -> A = 3.3
    # t=1: done, delta = 2 - 1.0 = 1.0                     -> A = 1.0 (no leak from t=2)
    # t=0: delta = 1 + 0.9*1.0 - 0.5 = 1.4, A = 1.4 + 0.72*1.0 = 2.12
    torch.testing.assert_close(advantages[:, 0], torch.tensor([2.12, 1.0, 3.3]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([2.62, 2.0, 4.8]))
    # env 1, values zero: A = discounted lambda-returns
    # t=2: 1 + 0.9*10 = 10; t=1: 0.72*10 = 7.2; t=0: 0.72*7.2 = 5.184
    torch.testing.assert_close(advantages[:, 1], torch.tensor([5.184, 7.2, 10.0]))


def test_gae_with_lambda_one_equals_discounted_monte_carlo_return() -> None:
    rewards = torch.tensor([[0.0], [0.0], [1.0], [5.0]])
    values = torch.rand(4, 1)
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    bootstrap = torch.tensor([10.0])
    _, returns = compute_gae(rewards, values, dones, bootstrap, gamma=0.9, gae_lambda=1.0)
    torch.testing.assert_close(returns[:, 0], torch.tensor([0.81, 0.9, 1.0, 5.0 + 9.0]))


def test_buffer_allocation_shapes_and_flatten() -> None:
    buffer = RolloutBuffer.allocate(7, 3, (2, 4, 5), torch.device("cpu"))
    assert buffer.observations.shape == (7, 3, 2, 4, 5)
    assert buffer.observations.dtype == torch.float32
    assert buffer.actions.shape == (7, 3) and buffer.actions.dtype == torch.int64
    for name in ("log_probs", "values", "rewards", "dones"):
        assert getattr(buffer, name).shape == (7, 3)
    assert (buffer.rollout_length, buffer.num_envs) == (7, 3)

    buffer.observations[2, 1] = 1.0
    zeros = torch.zeros(7, 3)
    batch = flatten_rollout(buffer, zeros, zeros)
    assert len(batch) == 21
    assert batch.observations.shape == (21, 2, 4, 5)
    assert batch.observations[2 * 3 + 1].sum() == 2 * 4 * 5  # (t, env) -> t * N + env


# --------------------------------------------------------------------------- PPO loss


def _loss_inputs(batch: int = 6, actions: int = 4):
    torch.manual_seed(0)
    logits = torch.randn(batch, actions, requires_grad=True)
    acts = torch.randint(0, actions, (batch,))
    old_log_probs = torch.distributions.Categorical(logits=logits.detach()).log_prob(acts)
    return logits, acts, old_log_probs


def test_ppo_loss_at_old_policy_is_minus_mean_advantage_with_no_kl() -> None:
    logits, actions, old_log_probs = _loss_inputs()
    advantages = torch.tensor([1.0, -2.0, 0.5, 3.0, 0.0, -1.0])
    values = torch.zeros(6)
    returns = torch.ones(6)
    terms = ppo_loss(logits, values, actions, old_log_probs, advantages, returns, 0.2, 0.5, 0.01)

    torch.testing.assert_close(terms.policy_loss, -advantages.mean())
    torch.testing.assert_close(terms.value_loss, torch.tensor(0.5))
    torch.testing.assert_close(terms.approx_kl, torch.tensor(0.0))
    assert terms.clip_fraction == 0.0
    expected_entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
    torch.testing.assert_close(terms.total, -advantages.mean() + 0.25 - 0.01 * expected_entropy)


def test_ppo_loss_clips_ratio_and_gradient_raises_good_actions() -> None:
    logits, actions, old_log_probs = _loss_inputs(batch=1)
    advantages = torch.tensor([1.0])
    # New policy far more likely than old: ratio > 1 + clip -> no gradient through policy term.
    shifted_old = old_log_probs - 1.0
    terms = ppo_loss(logits, torch.zeros(1), actions, shifted_old, advantages, torch.zeros(1), 0.2, 0.0, 0.0)
    assert terms.clip_fraction == 1.0
    torch.testing.assert_close(terms.policy_loss, torch.tensor(-1.2))
    terms.total.backward()
    assert torch.all(logits.grad == 0)

    # Unclipped with positive advantage: gradient descent increases that action's logit.
    logits.grad = None
    terms = ppo_loss(logits, torch.zeros(1), actions, old_log_probs, advantages, torch.zeros(1), 0.2, 0.0, 0.0)
    terms.total.backward()
    assert logits.grad[0, actions[0]] < 0


def test_ppo_loss_rejects_mismatched_value_shape() -> None:
    logits, actions, old_log_probs = _loss_inputs()
    with pytest.raises(ValueError):
        ppo_loss(logits, torch.zeros(6, 1), actions, old_log_probs, torch.zeros(6), torch.zeros(6), 0.2, 0.5, 0.0)


def test_normalize_and_explained_variance() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = normalize(x)
    assert abs(float(normalized.mean())) < 1e-6
    assert abs(float(normalized.std()) - 1.0) < 1e-5
    targets = torch.tensor([1.0, 2.0, 3.0])
    assert explained_variance(targets, targets) == 1.0
    assert explained_variance(torch.zeros(3), targets) == pytest.approx(1.0 - targets.var().item() / targets.var().item())
    assert math.isnan(explained_variance(torch.zeros(3), torch.ones(3)))


def test_linear_decay() -> None:
    assert linear_decay(1.0, 0, 4) == 1.0
    assert linear_decay(1.0, 3, 4) == 0.25


# --------------------------------------------------------------------------- collector


def test_collector_fills_buffer_with_aligned_transitions() -> None:
    torch.manual_seed(0)
    env = ScriptedVecEnv(num_envs=2)
    policy = MLPPolicy(env.observation_shape, env.num_actions)
    buffer = RolloutBuffer.allocate(4, 2, env.observation_shape, torch.device("cpu"))
    collector = RolloutCollector(env, buffer, env.reset())

    result = collector.collect(policy)

    # observations[t] is the observation the action at t was taken from.
    torch.testing.assert_close(buffer.observations[:, :, 0, 0, 0], torch.tensor([[0.0, 0.0], [1, 1], [2, 2], [3, 3]]))
    torch.testing.assert_close(buffer.rewards, torch.tensor([[0.0, 1.0], [10, 11], [20, 21], [30, 31]]))
    # dones[t] marks that the action at t ended the episode (counter hits 3 after step index 2).
    torch.testing.assert_close(buffer.dones, torch.tensor([[0.0, 0.0], [0, 0], [1, 1], [0, 0]]))
    np.testing.assert_array_equal(np.stack(env.received_actions), buffer.actions.numpy())

    with torch.no_grad():
        output = policy(buffer.observations.flatten(0, 1))
        expected_log_probs = torch.distributions.Categorical(logits=output.logits).log_prob(buffer.actions.flatten())
        torch.testing.assert_close(buffer.values.flatten(), output.value)
        torch.testing.assert_close(buffer.log_probs.flatten(), expected_log_probs)
        torch.testing.assert_close(result.bootstrap_value, policy(torch.full((2, 1, 1, 1), 4.0)).value)

    # A second rollout continues where the first stopped.
    collector.collect(policy)
    assert buffer.observations[0, 0, 0, 0, 0] == 4.0


def test_collector_restores_policy_mode_and_uses_eval_mode() -> None:
    class ModeRecorder(MLPPolicy):
        modes: list[bool] = []

        def forward(self, observations):
            self.modes.append(self.training)
            return super().forward(observations)

    env = ScriptedVecEnv(num_envs=1)
    policy = ModeRecorder(env.observation_shape, env.num_actions)
    policy.train()
    buffer = RolloutBuffer.allocate(2, 1, env.observation_shape, torch.device("cpu"))
    RolloutCollector(env, buffer, env.reset()).collect(policy)
    assert policy.modes and not any(policy.modes)
    assert policy.training


def test_collector_returns_finished_episodes() -> None:
    env = CorridorVecEnv(num_envs=3, length=2, max_steps=1)
    policy = MLPPolicy(env.observation_shape, env.num_actions)
    buffer = RolloutBuffer.allocate(5, 3, env.observation_shape, torch.device("cpu"))
    result = RolloutCollector(env, buffer, env.reset()).collect(policy)
    assert len(result.finished_episodes) == 15  # every episode lasts exactly one step
    assert torch.all(buffer.dones == 1.0)


# --------------------------------------------------------------------------- metrics & checkpoint


def test_episode_stats_window() -> None:
    window = EpisodeStatsWindow(window=2)
    assert window.summary() == {}
    window.add(EpisodeSummary(0, s, 10 * s, float(s), "hit_wall") for s in (1, 2, 3))
    assert window.summary() == {"episodes": 3.0, "mean_score": 2.5, "mean_length": 25.0, "mean_return": 2.5}


def test_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    policy = MLPPolicy((3, 1, 1), 4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy(torch.randn(2, 3, 1, 1)).value.sum().backward()
    optimizer.step()
    save_checkpoint(tmp_path / "sub" / "ckpt.pt", policy, optimizer, env_steps=1234)

    restored = MLPPolicy((3, 1, 1), 4)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    assert load_checkpoint(tmp_path / "sub" / "ckpt.pt", restored, restored_optimizer) == 1234
    for a, b in zip(policy.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(a, b)
    assert restored_optimizer.state_dict()["state"][0]["step"] == 1


def test_latest_checkpoint_picks_most_env_steps(tmp_path) -> None:
    assert latest_checkpoint(tmp_path / "missing") is None
    assert latest_checkpoint(tmp_path) is None
    for name in ("checkpoint_0000000999.pt", "checkpoint_0000001000.pt", "checkpoint_99.pt.bak", "other.pt"):
        (tmp_path / name).touch()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint_0000001000.pt"


# --------------------------------------------------------------------------- end to end


def _corridor_trainer(seed: int, total_env_steps: int, **overrides) -> Trainer:
    env = CorridorVecEnv(num_envs=8, length=5, max_steps=20)
    torch.manual_seed(seed)
    policy = MLPPolicy(env.observation_shape, env.num_actions)
    settings = dict(rollout_length=32, num_epochs=4, num_minibatches=4, learning_rate=3e-3, gamma=0.9, device="cpu")
    config = TrainConfig(total_env_steps=total_env_steps, seed=seed, **(settings | overrides))
    return Trainer(policy, env, config, logger=NullLogger())


def test_end_to_end_learning_on_corridor() -> None:
    trainer = _corridor_trainer(seed=1, total_env_steps=40_000)
    history = trainer.train()
    early = history[2]["mean_length"]
    late = history[-1]["mean_length"]
    # Optimal: 4 steps; random walk with 4 actions mostly fails within 20 steps.
    assert early > 10
    assert late < 5.0
    assert history[-1]["mean_return"] > 0.95
    assert trainer.env_steps == len(history) * 32 * 8


def test_training_is_deterministic_on_cpu() -> None:
    # Action sampling uses torch's global RNG (seeded by the Trainer), so runs
    # must not be interleaved.
    a = _corridor_trainer(seed=3, total_env_steps=2048)
    history_a = a.train()
    b = _corridor_trainer(seed=3, total_env_steps=2048)
    history_b = b.train()
    for x, y in zip(a.policy.state_dict().values(), b.policy.state_dict().values()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert [r["policy_loss"] for r in history_a] == [r["policy_loss"] for r in history_b]


def test_trainer_writes_checkpoints(tmp_path) -> None:
    trainer = _corridor_trainer(seed=0, total_env_steps=3 * 256, checkpoint_dir=tmp_path, checkpoint_interval=2)
    trainer.train()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint_0000000512.pt", "checkpoint_0000000768.pt"]


def test_resume_continues_counters_schedule_and_optimizer_state(tmp_path) -> None:
    steps_per_update = 32 * 8
    first = _corridor_trainer(seed=0, total_env_steps=4 * steps_per_update, checkpoint_dir=tmp_path)
    first.train()
    checkpoint = latest_checkpoint(tmp_path)
    assert checkpoint == tmp_path / "checkpoint_0000001024.pt"

    # A fresh trainer with a different initialisation and a doubled budget.
    second = _corridor_trainer(seed=5, total_env_steps=8 * steps_per_update, checkpoint_dir=tmp_path)
    assert resume_from_checkpoint(second, checkpoint) == 4 * steps_per_update
    assert second.env_steps == 4 * steps_per_update and second.completed_updates == 4
    for a, b in zip(first.policy.state_dict().values(), second.policy.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    first_state = first.optimizer.state_dict()["state"]
    second_state = second.optimizer.state_dict()["state"]
    assert second_state[0]["step"] == first_state[0]["step"] == 4 * 4 * 4  # updates * epochs * minibatches
    torch.testing.assert_close(second_state[0]["exp_avg"], first_state[0]["exp_avg"], rtol=0, atol=0)

    history = second.train()
    assert [record["update"] for record in history] == [5.0, 6.0, 7.0, 8.0]
    assert [record["env_steps"] for record in history] == [k * steps_per_update for k in (5, 6, 7, 8)]
    # LR follows the schedule over the new total: update index 4 of 8.
    assert history[0]["learning_rate"] == pytest.approx(linear_decay(3e-3, 4, 8))
    assert second.env_steps == 8 * steps_per_update
    assert second.optimizer.state_dict()["state"][0]["step"] == 8 * 4 * 4
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint_0000002048.pt"


def test_resume_at_or_past_total_trains_nothing(tmp_path) -> None:
    first = _corridor_trainer(seed=0, total_env_steps=2 * 256, checkpoint_dir=tmp_path)
    first.train()
    second = _corridor_trainer(seed=0, total_env_steps=2 * 256)
    resume_from_checkpoint(second, latest_checkpoint(tmp_path))
    assert second.train() == []
    assert second.env_steps == 512


# --------------------------------------------------------------------------- keeping checkpoints


def test_prune_checkpoints_keeps_the_given_one_and_ignores_other_files(tmp_path) -> None:
    names = ["checkpoint_0000000001.pt", "checkpoint_0000000002.pt", "checkpoint_0000000003.pt"]
    others = ["checkpoint_99.pt.bak", "other.pt", "config.toml", "metrics.jsonl"]
    for name in names + others:
        (tmp_path / name).touch()
    deleted = prune_checkpoints(tmp_path, keep=tmp_path / "checkpoint_0000000003.pt")
    assert sorted(path.name for path in deleted) == names[:2]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(["checkpoint_0000000003.pt", *others])
    assert prune_checkpoints(tmp_path / "missing", keep=tmp_path / "x.pt") == []


def test_trainer_without_keep_checkpoints_leaves_only_the_latest(tmp_path) -> None:
    (tmp_path / "notes.txt").touch()
    trainer = _corridor_trainer(
        seed=0, total_env_steps=4 * 256, checkpoint_dir=tmp_path, checkpoint_interval=1, keep_checkpoints=False
    )
    trainer.train()
    assert [path.name for path in list_checkpoints(tmp_path)] == ["checkpoint_0000001024.pt"]
    assert (tmp_path / "notes.txt").exists()


# --------------------------------------------------------------------------- forking


def test_fork_loads_parent_weights_and_optimizer_but_starts_its_own_counter(tmp_path) -> None:
    parent_dir, child_dir = tmp_path / "parent", tmp_path / "child"
    parent = _corridor_trainer(seed=0, total_env_steps=2 * 256, checkpoint_dir=parent_dir)
    parent.train()
    checkpoint = latest_checkpoint(parent_dir)
    parent_files = sorted(path.name for path in parent_dir.iterdir())

    child = _corridor_trainer(seed=7, total_env_steps=3 * 256, checkpoint_dir=child_dir, learning_rate=1e-3)
    assert fork_from_checkpoint(child, checkpoint) == 512
    assert child.env_steps == 0 and child.completed_updates == 0
    for a, b in zip(parent.policy.state_dict().values(), child.policy.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    parent_state = parent.optimizer.state_dict()["state"][0]
    torch.testing.assert_close(child.optimizer.state_dict()["state"][0]["exp_avg"], parent_state["exp_avg"], rtol=0, atol=0)

    history = child.train()
    assert [record["update"] for record in history] == [1.0, 2.0, 3.0]
    assert history[0]["learning_rate"] == pytest.approx(1e-3)  # the child's own LR schedule, from its start
    assert [path.name for path in list_checkpoints(child_dir)] == ["checkpoint_0000000768.pt"]
    assert sorted(path.name for path in parent_dir.iterdir()) == parent_files  # the parent is untouched
