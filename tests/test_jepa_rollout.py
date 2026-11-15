"""Rollout training of the jepa predictor (``model.rollout_steps``): where the step sequences come from and the loss by hand."""

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.schema import experiment_schema
from attentive_snake.factory import make_policy, make_trainer, make_vec_env
from attentive_snake.models.auxiliary import TransitionBatch, TransitionSequences, auxiliary_objective, sequence_length
from attentive_snake.models.blocks.heads import ValueHead
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.jepa.objective import PredictionObjective, cosine_distance, masked_mean, squared_distance
from attentive_snake.models.jepa.rollout import Rollout, steps_of, unroll
from attentive_snake.training import NullLogger
from attentive_snake.training.dqn.n_step import Horizon
from attentive_snake.training.dqn.prioritized_replay import PrioritizedReplayBuffer
from attentive_snake.training.dqn.replay_storage import ReplayStorage
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer
from attentive_snake.training.ppo.rollout import RolloutBuffer, flatten_rollout
from attentive_snake.training.ppo.sequences import rollout_sequences
from attentive_snake.training.slot_sequences import slot_sequences
from attentive_snake.training_metric import setting_is_on
from test_dqn_parts import CPU
from test_jepa import batch, make, tiny_dqn
from test_n_step import filled, vector_step

# ---------------------------------------------------------------- sequences from the DQN ring
# test_n_step's data: two envs, six vector steps. Observations are 10 * env + t, the next one + 1, actions = env.
# Env 0 dies (terminated) at its step 2, env 1 is cut by a limit (truncated) at its step 1.


def sequences(storage: ReplayStorage, slots: list[int], length: int = 3):
    """(next observations, valid, terminal) of each sequence, the columns that are not valid cut off."""
    found = storage.sequences(np.array(slots), length, CPU)
    rows = []
    for row in range(len(slots)):
        steps = int(found.valid[row].sum())
        assert found.valid[row].tolist() == [1.0] * steps + [0.0] * (length - steps)  # once a step is missing, the rest is
        rows.append((found.next_observations[row, :steps, 0].tolist(), found.terminal[row, :steps].tolist()))
    return rows


def test_ring_sequences_follow_one_env_and_stop_at_an_episode_end_and_at_the_newest_step() -> None:
    storage = filled(capacity=12)  # nothing overwritten: step t of env e sits in slot 2 * t + e
    assert sequences(storage, [0, 2, 4]) == [
        ([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]),  # env 0 from t = 0: three steps, the last one dies
        ([2.0, 3.0], [0.0, 1.0]),  # from t = 1: it dies after two; t = 3 is another episode
        ([3.0], [1.0]),  # the dying step alone: its own next observation is still real
    ]
    assert sequences(storage, [6, 8, 10]) == [([4.0, 5.0, 6.0], [0.0] * 3), ([5.0, 6.0], [0.0] * 2), ([6.0], [0.0])]  # t = 5 is the newest
    assert sequences(storage, [1, 3, 5]) == [([11.0, 12.0], [0.0, 1.0]), ([12.0], [1.0]), ([13.0, 14.0, 15.0], [0.0] * 3)]  # env 1, cut at t = 1
    found = storage.sequences(np.array([0, 1]), 3, CPU)
    assert found.actions.tolist() == [[0, 0, 0], [1, 1, 1]] and found.actions.dtype == torch.int64  # each env's own actions
    assert found.next_observations.shape == (2, 3, 1) and found.valid.dtype == torch.float32


def test_a_truncation_that_is_bootstrapped_still_ends_the_sequence_but_is_not_terminal() -> None:
    storage = filled(capacity=12, bootstrap_truncated=True)
    assert sequences(storage, [1]) == [([11.0, 12.0], [0.0, 0.0])]


def test_ring_sequences_across_the_wrap_around() -> None:
    storage = filled(capacity=8)  # holds t = 2..5: t = 4 in slots 0-1, t = 5 in 2-3, t = 2 in 4-5, t = 3 in 6-7
    assert sequences(storage, [6, 7]) == [([4.0, 5.0, 6.0], [0.0] * 3), ([14.0, 15.0, 16.0], [0.0] * 3)]  # t = 3, 4, 5 over the ring's end
    assert sequences(storage, [2, 3]) == [([6.0], [0.0]), ([16.0], [0.0])]  # the newest step: the slots after it hold t = 2, the oldest
    assert sequences(storage, [4]) == [([3.0], [1.0])]  # env 0's death at t = 2
    assert sequences(storage, [0], length=1) == [([5.0], [0.0])]


def test_a_ring_whose_capacity_is_no_multiple_of_the_envs_still_finds_the_later_steps() -> None:
    storage = filled(capacity=7)  # 12 writes into 7 slots: t = 5 in slots 3-4, t = 4 in 1-2, env 1's t = 3 in slot 0, env 0's t = 2... overwritten
    assert sequences(storage, [0, 1, 2]) == [([14.0, 15.0, 16.0], [0.0] * 3), ([5.0, 6.0], [0.0] * 2), ([15.0, 16.0], [0.0] * 2)]


def test_sequences_need_one_transition_per_env_in_every_write() -> None:
    storage = ReplayStorage(8, (1,), Horizon(1, 0.9))
    step = vector_step(0, False)
    storage.write(step)
    one = dataclasses.replace(step, **{name: getattr(step, name)[:1] for name in ("observations", "actions", "rewards", "next_observations", "dones", "ends")})
    storage.write(one)  # allowed for plain one-step replay ...
    with pytest.raises(ValueError, match="same number"):
        storage.sequences(np.array([0]), 2, CPU)  # ... but a later step of an env can no longer be found


@pytest.mark.parametrize("prioritized", [False, True])
def test_both_buffers_hand_out_the_sequences_of_their_storage(prioritized) -> None:
    rng = np.random.default_rng(0)
    horizon = Horizon(1, 0.9)
    buffer = PrioritizedReplayBuffer(12, (1,), rng, 0.6, 1e-6, lambda: 0.4, horizon) if prioritized else UniformReplayBuffer(12, (1,), rng, horizon)
    for t in range(6):
        buffer.add(vector_step(t, False))
    sample = buffer.sample(5, CPU)
    found = buffer.sequences(sample.indices, 3, CPU)
    assert torch.equal(found.actions[:, 0], sample.actions) and torch.equal(found.next_observations[:, 0], sample.next_observations)
    assert torch.equal(found.terminal[:, 0], sample.dones) and found.valid[:, 0].tolist() == [1.0] * 5


def test_slot_sequences_repeat_the_last_present_slot_where_a_step_is_missing() -> None:
    ends = np.array([False, False, True, False, False, False])
    found = slot_sequences(ends, starts=np.array([0, 1]), later_steps=np.array([2, 2]), stride=2, length=3)
    assert found.slots.tolist() == [[0, 2, 2], [1, 3, 5]] and found.present.tolist() == [[True, True, False], [True, True, True]]
    with pytest.raises(ValueError):
        slot_sequences(ends, np.array([0]), np.array([0]), 2, 0)


# ---------------------------------------------------------------- sequences from a PPO rollout


def test_ppo_sequences_stop_at_an_episode_end_and_at_the_end_of_the_rollout() -> None:
    buffer = RolloutBuffer.allocate(4, 2, (1,), CPU, keep_next_observations=True)
    for t in range(4):
        buffer.actions[t] = torch.tensor([t, 10 + t])
        buffer.next_observations[t] = torch.tensor([[t + 1.0], [10 + t + 1.0]])
    buffer.dones[1, 0] = 1.0  # env 0's episode ends with its step 1
    found = rollout_sequences(buffer, 3)  # rows as flatten_rollout orders them: step 0 of both envs, step 1 of both, ...
    assert found.valid.tolist() == [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]]
    assert found.next_observations[0, :2, 0].tolist() == [1.0, 2.0] and found.terminal[0, :2].tolist() == [0.0, 1.0]
    assert found.next_observations[1, :, 0].tolist() == [11.0, 12.0, 13.0] and found.actions[1].tolist() == [10, 11, 12]
    assert found.actions[4].tolist()[:2] == [2, 3]  # env 0 after its episode end: the new episode's steps 2 and 3
    zeros = torch.zeros(4, 2)
    flat = flatten_rollout(buffer, zeros, zeros, found)
    picked = flat.select(torch.tensor([3, 0]))
    assert torch.equal(picked.sequences.actions[:, 0], picked.actions) and torch.equal(picked.sequences.next_observations[:, 0], picked.next_observations)
    with pytest.raises(ValueError, match="next observations"):
        rollout_sequences(RolloutBuffer.allocate(4, 2, (1,), CPU), 3)


# ---------------------------------------------------------------- the rollout loss by hand


class Scale(nn.Module):
    """Encoder: the observation times a weight of 1 (so it has something to train)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations * self.weight


class AddAction(nn.Module):
    """Predictor: the next embedding is the embedding plus the action's number (times a gain of 1)."""

    def __init__(self, num_actions: int = 2) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.gain = nn.Parameter(torch.ones(()))

    def step(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return embeddings + self.gain * actions.unsqueeze(-1).float()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # (B, 1) -> (B, A, 1)
        return torch.stack([self.step(embeddings, torch.full((len(embeddings),), action)) for action in range(self.num_actions)], dim=1)


def doubling_decoder() -> ValueHead:
    decoder = ValueHead(1)
    with torch.no_grad():
        decoder.linear.weight.fill_(2.0)
        decoder.linear.bias.zero_()
    return decoder


def hand_batch() -> TransitionBatch:
    """Two rollouts of three steps from s_0 = 1 and s_0 = 10; one-number observations, so every term can be read off.

    Row 0: actions 1, 0, 1 -> predictions 2, 2, 3; the game really went 2, 3, 3 and ended with the third step.
    Row 1: actions 1, 1, 0 -> predictions 11, 12, 12; really 11, 13, then the episode had ended: the third step is not valid.
    """
    steps = TransitionSequences(
        actions=torch.tensor([[1, 0, 1], [1, 1, 0]]),
        next_observations=torch.tensor([[[2.0], [3.0], [3.0]], [[11.0], [13.0], [99.0]]]),
        valid=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        terminal=torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    return TransitionBatch(torch.tensor([[1.0], [10.0]]), steps.actions[:, 0], steps.next_observations[:, 0], steps.valid[:, 0], steps)


def hand_objective(value_coefficient: float, trains_encoder: bool = True, coefficient: float = 2.0):
    encoder, predictor, decoder = Scale(), AddAction(), doubling_decoder()
    objective = PredictionObjective(
        encoder, predictor, coefficient, 0.99, squared_distance, trains_encoder=trains_encoder, value_decoder=decoder, rollout=Rollout(3, value_coefficient)
    )
    return objective, encoder, predictor, decoder


def test_unroll_feeds_the_predictor_its_own_output() -> None:
    predictions = unroll(AddAction(), torch.tensor([[1.0], [10.0]]), hand_batch().sequences.actions)
    assert predictions.squeeze(-1).tolist() == [[2.0, 2.0, 3.0], [11.0, 12.0, 12.0]]


def test_the_rollout_loss_by_hand_with_a_masked_step_after_an_episode_end() -> None:
    objective, *_ = hand_objective(value_coefficient=0.0)
    terms = objective.loss(hand_batch())
    # squared distances to the real states: row 0: 0, 1, 0; row 1: 0, 1, (masked) -> mean over the 5 valid steps = 0.4
    assert float(terms.loss.detach()) == pytest.approx(2.0 * 0.4)
    assert terms.metrics["prediction_loss"] == pytest.approx(0.4)
    assert terms.metrics["rollout_last_loss"] == pytest.approx(0.0)  # the third step: only row 0 has one, and it is exact
    assert terms.metrics["value_gap"] == pytest.approx(0.8)  # |2 * (prediction - real)|: 0, 2, 0, 0, 2 over 5
    assert "chained_value_gap" not in terms.metrics and objective.sequence_length == 3


def test_the_chained_value_term_by_hand_skips_states_the_game_did_not_go_on_from() -> None:
    objective, *_ = hand_objective(value_coefficient=0.5)
    terms = objective.loss(hand_batch())
    # Q(z, a') = 2 * (z + a'): the gap to the real state's Q is 2 * (prediction - real) for both actions.
    # It counts where the step is valid and the real state is not terminal: row 0 steps 1, 2 (gaps 0, -2), row 1 step 1 (gap 0).
    # Huber(2) = 1.5 -> (0 + 1.5 + 0) / 3 = 0.5; the mean absolute gap is (0 + 2 + 0) / 3.
    assert float(terms.loss.detach()) == pytest.approx(2.0 * (0.4 + 0.5 * 0.5))
    assert terms.metrics["chained_value_gap"] == pytest.approx(2.0 / 3.0)


def gradients(value_coefficient: float, trains_encoder: bool = True):
    objective, encoder, predictor, decoder = hand_objective(value_coefficient, trains_encoder)
    objective.loss(hand_batch()).loss.backward()
    return encoder.weight.grad, predictor.gain.grad, decoder.linear.weight.grad


def test_the_chained_value_term_trains_the_predictor_only() -> None:
    encoder_without, predictor_without, _ = gradients(0.0)
    encoder_with, predictor_with, decoder_with = gradients(0.5)
    assert encoder_without is not None and torch.equal(encoder_with, encoder_without)  # nothing of it reaches the encoder
    assert decoder_with is None  # nor the decoder's weights
    # The embedding it reads is not moved either: only the gain of the predictor's step *from* the prediction gets a gradient,
    # d/d gain of mean Huber(2 * (z + gain * a' - ...)) over actions = 0.5 * (-1 * 2 * 1) for the one row with a gap, / 3 rows.
    assert float(predictor_with - predictor_without) == pytest.approx(2.0 * 0.5 * (0.5 * -2.0) / 3.0)
    assert gradients(0.5, trains_encoder=False)[0] is None


def test_the_target_encoder_and_real_values_carry_no_gradient_and_all_invalid_steps_give_zero() -> None:
    objective, *_ = hand_objective(0.5)
    blank = hand_batch()
    nothing = dataclasses.replace(blank, sequences=dataclasses.replace(blank.sequences, valid=torch.zeros(2, 3)))
    assert float(objective.loss(nothing).loss.detach()) == 0.0
    objective.loss(blank).loss.backward()
    assert all(parameter.grad is None for parameter in objective.target.module.parameters())


# ---------------------------------------------------------------- one step is today's loss


@pytest.mark.parametrize("kind_and_spec", [("jepa", ("vector", 1)), ("jepa", ("grid", 1)), ("jepa_tokens", ("grid", 1))])
@pytest.mark.parametrize("value_training", ["predicted", "actual"])
def test_one_rollout_step_is_exactly_the_one_step_loss(kind_and_spec, value_training) -> None:
    kind, spec = kind_and_spec
    policy = make(spec, kind=kind, value_training=value_training, prediction_trains_encoder=True, num_heads=2, mlp_ratio=1)
    objective = auxiliary_objective(policy)
    assert sequence_length(objective) == 1
    transitions = batch(policy)
    transitions = dataclasses.replace(transitions, valid=torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]))
    terms = objective.loss(transitions)

    # the loss as it was before rollouts existed
    predictions = policy.predictor.step(policy.encode(transitions.observations), transitions.actions)
    targets = objective.target(transitions.next_observations)
    distance = cosine_distance if value_training == "predicted" else squared_distance
    expected = masked_mean(distance(predictions, targets), transitions.valid)
    decoder, read = policy.net["value_decoder"], policy.readout
    with torch.no_grad():
        gaps = decoder(read(predictions)) - decoder(read(policy.encode(transitions.next_observations)))
    if value_training == "actual":
        expected = expected + masked_mean(gaps.pow(2), transitions.valid)
    assert torch.equal(terms.loss, policy.config.prediction_coef * expected)
    assert set(terms.metrics) == {"prediction_loss", "embedding_std", "value_gap"}
    assert terms.metrics["value_gap"] == pytest.approx(float(masked_mean(gaps.abs(), transitions.valid)), rel=1e-6)

    # and sequences of one step are that same batch
    again = objective.loss(dataclasses.replace(transitions, sequences=steps_of(transitions)))
    assert torch.equal(again.loss, terms.loss) and again.metrics == terms.metrics


def test_rollout_settings_are_validated_and_never_change_the_network() -> None:
    with pytest.raises(ValueError, match="rollout_steps"):
        ModelConfig(kind="jepa", rollout_steps=0)
    with pytest.raises(ValueError, match="rollout_value_coef"):
        ModelConfig(kind="jepa", rollout_value_coef=-1.0)
    plain, rolled = make(), make(rollout_steps=3, rollout_value_coef=0.5)
    assert list(plain.state_dict()) == list(rolled.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), rolled.state_dict().values()))
    assert sequence_length(auxiliary_objective(rolled)) == 3 and sequence_length(object()) == 1


# ---------------------------------------------------------------- through the learners


def trained_metrics(config) -> set[str]:
    env = make_vec_env(config)
    return set().union(*make_trainer(config, make_policy(config, env), env, NullLogger()).train())


@pytest.mark.parametrize("kind", ["jepa", "jepa_tokens"])
def test_dqn_trains_the_rollout_and_the_chained_values(kind) -> None:
    model = {"kind": kind, "rollout_steps": 3, "num_heads": 2, "mlp_ratio": 1}
    assert {"prediction_loss", "rollout_last_loss", "chained_value_gap"} <= trained_metrics(tiny_dqn(model))
    assert not {"rollout_last_loss", "chained_value_gap"} & trained_metrics(tiny_dqn({"kind": kind, "num_heads": 2, "mlp_ratio": 1}))  # one step: as before


def test_the_chained_values_are_left_out_when_switched_off_and_with_the_dueling_head() -> None:
    off = trained_metrics(tiny_dqn({"rollout_steps": 3, "rollout_value_coef": 0.0}))
    assert "rollout_last_loss" in off and "chained_value_gap" not in off
    dueling = tiny_dqn({"rollout_steps": 3})
    dueling = dataclasses.replace(dueling, algorithm_settings=dataclasses.replace(dueling.algorithm_settings, dueling=True))
    logged = trained_metrics(dueling)
    assert "rollout_last_loss" in logged and "chained_value_gap" not in logged  # its value decoder reads the current state, not Q


def test_dqn_with_n_step_and_prioritized_replay_still_gets_its_sequences() -> None:
    config = tiny_dqn({"rollout_steps": 2})
    config = dataclasses.replace(config, algorithm_settings=dataclasses.replace(config.algorithm_settings, n_step=3, prioritized=True))
    assert {"rollout_last_loss", "chained_value_gap", "priority_mean"} <= trained_metrics(config)


def test_ppo_trains_the_rollout_from_its_rollout_buffer_without_the_chained_values() -> None:
    config = experiment_from_tables({
        "game": {"width": 4, "height": 4, "max_steps_without_food": 8},
        "model": {"kind": "jepa", "d_model": 8, "num_layers": 1, "conv_channels": 4, "conv_layers": 1, "prediction_coef": 1.0, "rollout_steps": 3},
        "train": {"total_env_steps": 64, "device": "cpu", "log_interval": 32},
        "ppo": {"rollout_length": 8, "num_epochs": 1, "num_minibatches": 2},
        "env": {"num_envs": 4},
    })
    logged = trained_metrics(config)
    assert {"prediction_loss", "rollout_last_loss"} <= logged and "chained_value_gap" not in logged  # PPO's decoder gives no Q-values


# ---------------------------------------------------------------- what the app is told


def test_the_rollout_metrics_are_switched_on_above_one_step() -> None:
    assert not setting_is_on(1, above=1.0) and setting_is_on(3, above=1.0) and setting_is_on(1) and not setting_is_on(0)
    kinds = experiment_schema("snake")["model_kinds"]
    for kind in ("jepa", "jepa_attention", "jepa_tokens", "jepa_temporal"):
        rollout = [metric for metric in kinds[kind]["metrics"] if metric["setting"] == "model.rollout_steps"]
        assert [metric["key"] for metric in rollout] == ["rollout_last_loss", "chained_value_gap"] and all(metric["above"] == 1.0 for metric in rollout)
        assert all(metric["above"] == 0.0 for metric in kinds[kind]["metrics"] if metric["setting"] == "model.prediction_coef")
        assert {"rollout_steps", "rollout_value_coef"} <= set(kinds[kind]["fields"])
    assert not any({"rollout_steps", "rollout_value_coef"} & set(info["fields"]) for kind, info in kinds.items() if not kind.startswith("jepa"))
