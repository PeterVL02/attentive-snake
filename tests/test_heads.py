"""Every model kind with either head, and the acting contract that reads both outputs."""

import pytest
import torch

from attentive_snake.models.acting import action_readout, can_sample, greedy_actions, sampled_actions
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.heads import HEADS
from attentive_snake.models.registry import POLICY_KINDS, build_policy
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.observation_spec import ObservationSpec
from attentive_snake.viz import diagram

A = 4
SMALL = dict(d_model=16, num_heads=2, num_layers=2, conv_channels=8, conv_layers=2)
SPECS = {
    "grid": ObservationSpec("grid", (2 * (3 + A), 5, 6), 2, A),
    "vector": ObservationSpec("vector", (2 * (5 + A),), 2, A),
}


def make(kind: str, observation: str, head: str):
    torch.manual_seed(0)
    return build_policy(ModelConfig(kind=kind, **SMALL), SPECS[observation], A, head)


@pytest.mark.parametrize("observation", sorted(SPECS))
@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_every_kind_builds_with_either_head_on_the_same_trunk(kind: str, observation: str) -> None:
    actor_critic, q = make(kind, observation, "actor_critic"), make(kind, observation, "q_values")
    obs = torch.rand(3, *SPECS[observation].shape)
    out = actor_critic(obs)
    assert isinstance(out, PolicyOutput) and out.logits.shape == (3, A) and out.value.shape == (3,)
    q_out = q(obs)
    assert isinstance(q_out, QValues) and q_out.q_values.shape == (3, A)

    forms = type(q).heads  # the kind's head forms (models.heads.HEADS, or its own with the same names)
    ac_names, q_names = list(forms["actor_critic"].modules(16, A)), list(forms["q_values"].modules(16, A))
    ac_keys, q_keys = set(actor_critic.state_dict()), set(q.state_dict())
    assert {key for key in ac_keys if key.split(".")[1] not in ac_names} == {key for key in q_keys if key.split(".")[1] not in q_names}  # the same trunk
    assert list(q.net)[-len(q_names):] == q_names and list(actor_critic.net)[-len(ac_names):] == ac_names
    if forms is HEADS:
        assert {key for key in q_keys if key.startswith("net.q_head.")} == {"net.q_head.linear.weight", "net.q_head.linear.bias"}
        assert "QHead" in diagram(q) and "PolicyHead" not in diagram(q)
    if hasattr(q, "attention_maps"):  # attention needs nothing from the head
        assert len(q.attention_maps(obs[:1])) == SMALL["num_layers"]


def test_an_unknown_head_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown policy head 'dueling'"):
        make("mlp", "vector", "dueling")
    assert set(HEADS) == {"actor_critic", "q_values"}


def test_greedy_actions_take_the_argmax_of_logits_or_q_values() -> None:
    scores = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    assert greedy_actions(PolicyOutput(scores, torch.zeros(2))).tolist() == [1, 0]
    assert greedy_actions(QValues(scores)).tolist() == [1, 0]
    with pytest.raises(TypeError):
        greedy_actions(scores)


def test_sampling_draws_from_the_policy_but_is_greedy_for_q_values() -> None:
    torch.manual_seed(0)
    certain = PolicyOutput(torch.tensor([[-50.0, 50.0, -50.0]]), torch.zeros(1))
    assert can_sample(certain) and sampled_actions(certain).tolist() == [1]
    q = QValues(torch.tensor([[0.0, 1.0, 5.0]]))
    assert not can_sample(q) and [int(sampled_actions(q)) for _ in range(5)] == [2] * 5


def test_readouts_show_probabilities_and_value_or_q_values() -> None:
    readout = action_readout(PolicyOutput(torch.tensor([[0.0, 0.0]]), torch.tensor([1.5])))
    assert (readout.kind, readout.values, readout.state_value) == ("probabilities", [0.5, 0.5], 1.5)
    q = action_readout(QValues(torch.tensor([[1.0, -2.0]])))
    assert q.to_dict() == {"kind": "q_values", "label": "Q-values", "values": [1.0, -2.0], "state_value": None}
