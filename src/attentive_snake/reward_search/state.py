"""What a reward search remembers: its spec, its generations and their members, and its status, as immutable data and JSON.

::

    {"version": 1, "spec": {...}, "objective": "mean_score", "higher_is_better": true,
     "start_reward": {"food": 1.0, ...}, "status": "running", "error": null, "created_at": 1763300000.0,
     "generations": [{"index": 0, "parent": "base", "checkpoint": "checkpoint_0000100000.pt", "winner": null,
                      "members": [{"index": 0, "run": "s-g0-m0", "reward": {...}, "state": "training",
                                   "score": null, "checkpoint": null, "resumed": false, "error": null}, ...]}]}

Member states: ``planned`` (written down, maybe not yet in the training queue), ``waiting`` (in the
queue), ``training``, then ``done`` (its final evaluation was read: ``score``, ``checkpoint``) or
``failed`` (``error`` says why). Search status: ``running``, ``finished`` (all generations judged),
``stopped`` (by the user) or ``failed`` (``error``).
"""

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from attentive_snake.reward_search.spec import SearchSpec, spec_from_json, spec_to_json

VERSION = 1
MemberState = Literal["planned", "waiting", "training", "done", "failed"]
SearchStatus = Literal["running", "finished", "stopped", "failed"]
ENDED_MEMBER_STATES = ("done", "failed")


@dataclass(frozen=True)
class Member:
    """
    Attributes:
        index: 0 keeps the generation's starting reward; the others train on a changed copy.
        run: Its run's name (of the search's game).
        reward: Its whole ``[reward]`` table.
        state: See the module docstring.
        score: The objective's value in its final evaluation (``None`` until ``done``, or if not available).
        checkpoint: The checkpoint that final evaluation judged (the next generation forks it if this member wins).
        resumed: It ended once without its final evaluation and was resumed (it is not resumed twice).
        error: Why it failed.
    """

    index: int
    run: str
    reward: dict[str, Any]
    state: MemberState = "planned"
    score: float | None = None
    checkpoint: str | None = None
    resumed: bool = False
    error: str | None = None

    @property
    def ended(self) -> bool:
        return self.state in ENDED_MEMBER_STATES


@dataclass(frozen=True)
class Generation:
    """
    Attributes:
        index: Counted from 0.
        parent: The run its members fork (the search's parent, then the previous winner).
        checkpoint: The parent's checkpoint they fork (``None``: its latest when each member starts).
        members: In index order.
        winner: Index of the winning member once all have ended.
    """

    index: int
    parent: str
    checkpoint: str | None
    members: tuple[Member, ...]
    winner: int | None = None

    @property
    def ended(self) -> bool:
        return all(member.ended for member in self.members)


@dataclass(frozen=True)
class SearchState:
    """
    Attributes:
        spec: What the search is.
        objective: Key of the statistic that judges the members.
        higher_is_better: Its direction.
        start_reward: The parent's whole ``[reward]`` table: generation 0's starting table and the
            reference of every bound (``perturbation``).
        generations: Planned so far, in order.
        status: See the module docstring.
        error: Why it failed.
        created_at: Seconds since the epoch.
    """

    spec: SearchSpec
    objective: str
    higher_is_better: bool
    start_reward: dict[str, Any]
    generations: tuple[Generation, ...] = ()
    status: SearchStatus = "running"
    error: str | None = None
    created_at: float = field(default_factory=time.time)


def state_to_json(state: SearchState) -> dict[str, Any]:
    data = dataclasses.asdict(state)
    return {"version": VERSION, **data, "spec": spec_to_json(state.spec)}


def state_from_json(data: dict[str, Any]) -> SearchState:
    """The state written by ``state_to_json`` (``KeyError`` / ``TypeError`` / ``ValueError`` otherwise)."""
    generations = tuple(
        Generation(**{**generation, "members": tuple(Member(**member) for member in generation["members"])})
        for generation in data["generations"]
    )
    fields = {key: value for key, value in data.items() if key not in ("version", "spec", "generations")}
    return SearchState(spec=spec_from_json(data["spec"]), generations=generations, **fields)
