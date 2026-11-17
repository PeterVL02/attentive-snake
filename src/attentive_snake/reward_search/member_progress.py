"""One member's next state, from what the backend says about its run now; queues its fork or one resume when due.

Idempotent and restart-safe: a ``planned`` member whose run already exists or waits (it was
queued just before the app stopped) is not queued again, and an ended member is judged from its
run directory, not from anything remembered.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

from attentive_snake.reward_search.backend import LaunchRefused, TrainingBackend
from attentive_snake.reward_search.member_config import member_tables
from attentive_snake.reward_search.state import Generation, Member, SearchState

_GONE = "its run was never created: its queue entry was removed or could not start (see the Queue tab)."
_NO_EVALUATION = "it ended without its final evaluation, also after one resume (see its log in the Runs tab)."


def _description(state: SearchState, generation: Generation, member: Member) -> str:
    changes = ", ".join(f"{name} = {member.reward[name]:g}" for name in state.spec.fields)
    role = "unchanged reward" if member.index == 0 else "changed reward"
    return f"Reward search {state.spec.name}, generation {generation.index}, member {member.index} ({role}): {changes}."


def advance_member(
    state: SearchState, generation: Generation, member: Member, backend: TrainingBackend, parent_tables: Mapping[str, Any] | None
) -> Member:
    """``member``'s state after looking at its run once. ``parent_tables``: the generation's parent's tables (for queueing)."""
    if member.ended:
        return member
    game = state.spec.game
    progress = backend.progress(game, member.run)
    if progress == "missing" and member.state == "planned":
        return _queue_fork(state, generation, member, backend, parent_tables)
    if progress == "missing":
        return dataclasses.replace(member, state="failed", error=_GONE)
    if progress in ("waiting", "training"):
        return dataclasses.replace(member, state=progress)
    return _judge(state, member, backend)


def _queue_fork(
    state: SearchState, generation: Generation, member: Member, backend: TrainingBackend, parent_tables: Mapping[str, Any] | None
) -> Member:
    if parent_tables is None:
        return dataclasses.replace(member, state="failed", error=f"the parent run {generation.parent!r} cannot be read.")
    tables = member_tables(parent_tables, member.reward, state.spec.settings)
    description = _description(state, generation, member)
    try:
        backend.enqueue_fork(state.spec.game, generation.parent, generation.checkpoint, member.run, tables, description)
    except LaunchRefused as error:
        return dataclasses.replace(member, state="failed", error=f"it could not be queued: {error}")
    return dataclasses.replace(member, state="waiting")


def _judge(state: SearchState, member: Member, backend: TrainingBackend) -> Member:
    """An ended run: done with its score if its final evaluation exists, else resumed once, else failed.

    A member evaluates every ``steps_per_member`` env steps, which is also its length, so the only
    evaluation it makes while training is the one at its last update (a run stops on a whole update,
    which may be a little short of the length): any evaluation made while training is the final one.
    """
    final = backend.last_training_evaluation(state.spec.game, member.run)
    if final is not None:
        score = final.statistics.get(state.objective)
        return dataclasses.replace(member, state="done", score=None if score is None else float(score), checkpoint=final.checkpoint)
    if member.resumed:
        return dataclasses.replace(member, state="failed", error=_NO_EVALUATION)
    try:
        backend.enqueue_resume(state.spec.game, member.run)
    except LaunchRefused as error:
        return dataclasses.replace(member, state="failed", error=f"it ended early and could not be resumed: {error}")
    return dataclasses.replace(member, state="waiting", resumed=True)
