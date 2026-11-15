"""The temporal_transformer kind's net children, state-dict keys, shapes, seeded initial weights and outputs, pinned.

Recorded before jepa_temporal reused its token construction: saved runs must keep loading, seed 0
must keep building the same network, and that network must keep computing the same outputs.
Grid and vector observations, both heads, with and without action tokens (history 3).
Re-record (only after a deliberate change): ``uv run python tests/test_temporal_transformer_pin.py``.
"""

import json
from pathlib import Path

import pytest
import torch

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.registry import build_policy
from attentive_snake.models.types import PolicyOutput
from attentive_snake.observation_spec import ObservationSpec

A, K = 4, 3
BASE = {"grid": 3, "vector": 6}
CONFIG = ModelConfig(kind="temporal_transformer", d_model=16, num_heads=2, num_layers=2)
DATA = Path(__file__).parent / "data" / "temporal_transformer_pins.json"
ENTRIES = [f"{o}/{h}/{a}" for o in ("grid", "vector") for h in ("actor_critic", "q_values") for a in ("actions", "no_actions")]


def spec(observation: str, actions: bool) -> ObservationSpec:
    channels = K * (BASE[observation] + (A if actions else 0))
    shape = (channels, 5, 5) if observation == "grid" else (channels,)
    return ObservationSpec(observation, shape, K, A if actions else 0)


def observations(observation_spec: ObservationSpec) -> torch.Tensor:
    """Seeded inputs; with action channels, frame 0 has no action and frame j > 0 action (b + j) % A."""
    obs = torch.rand(5, *observation_spec.shape, generator=torch.Generator().manual_seed(1))
    if observation_spec.has_actions:
        f, base = observation_spec.frame_channels, observation_spec.base_channels
        for b in range(obs.shape[0]):
            for j in range(K):
                obs[b, j * f + base : (j + 1) * f] = 0
                if j > 0:
                    obs[b, j * f + base + (b + j) % A] = 1
    return obs


def build(entry: str):
    observation, head, actions = entry.split("/")
    observation_spec = spec(observation, actions == "actions")
    torch.manual_seed(0)
    return build_policy(CONFIG, observation_spec, A, head).eval(), observations(observation_spec)


def outputs(policy, obs) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = policy(obs)
    return {"logits": out.logits, "value": out.value} if isinstance(out, PolicyOutput) else {"q_values": out.q_values}


def summary(tensor: torch.Tensor) -> list:
    tensor = tensor.double()
    return [list(tensor.shape), float(tensor.sum()), float(tensor.abs().sum())]


def record() -> dict:
    pins = {}
    for entry in ENTRIES:
        policy, obs = build(entry)
        pins[entry] = {
            "children": list(policy.net),
            "state_dict": {key: summary(value) for key, value in policy.state_dict().items()},
            "outputs": {key: summary(value) for key, value in outputs(policy, obs).items()},
        }
    return pins


RECORDED = json.loads(DATA.read_text()) if DATA.exists() else {}


@pytest.mark.parametrize("entry", ENTRIES)
def test_temporal_transformer_keys_shapes_seeded_weights_and_outputs_never_change(entry):
    policy, obs = build(entry)
    pinned = RECORDED[entry]
    state = policy.state_dict()
    assert list(state) == list(pinned["state_dict"])
    for key, (shape, total, absolute) in pinned["state_dict"].items():
        tensor = state[key].double()
        assert list(tensor.shape) == shape, key
        assert float(tensor.sum()) == pytest.approx(total, abs=1e-9) and float(tensor.abs().sum()) == pytest.approx(absolute, rel=1e-12), key
    assert list(policy.net) == pinned["children"]  # the diagram and the checkpoint key prefixes
    produced = outputs(policy, obs)
    assert set(produced) == set(pinned["outputs"])
    for key, (shape, total, absolute) in pinned["outputs"].items():
        tensor = produced[key].double()
        assert list(tensor.shape) == shape, key
        assert float(tensor.sum()) == pytest.approx(total, abs=1e-6) and float(tensor.abs().sum()) == pytest.approx(absolute, rel=1e-6), key


if __name__ == "__main__":
    DATA.write_text(json.dumps(record(), indent=2) + "\n")
