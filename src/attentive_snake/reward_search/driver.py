"""One step of a reward search: plan the next generation, or look at the current one's members, or judge it.

``advance`` is safe to call at any time and any number of times: it only acts on what the state and
the backend say now. A generation is planned (and saved by the caller) one step before its members
are queued, so a restart between the two queues nothing twice (``member_progress``).
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from attentive_snake.reward_search.backend import TrainingBackend
from attentive_snake.reward_search.generation import plan_generation
from attentive_snake.reward_search.member_progress import advance_member
from attentive_snake.reward_search.selection import winner
from attentive_snake.reward_search.state import Generation, SearchState

type Accept = Callable[[Mapping[str, Any]], bool]


def advance(state: SearchState, backend: TrainingBackend, accept: Accept) -> SearchState:
    """``state`` after one step; unchanged if the search is not running or nothing happened.

    ``accept(table)`` says whether the game takes a reward table (``reward_fields.valid_reward``).
    """
    if state.status != "running":
        return state
    if not state.generations:
        return _planned(state, state.spec.parent, state.spec.checkpoint, state.start_reward, accept)
    generation = state.generations[-1]
    if not generation.ended:
        generation = _looked_at(state, generation, backend)
        state = _with_last(state, generation)
        if not generation.ended:
            return state
    return _judged(state, generation, accept)


def _looked_at(state: SearchState, generation: Generation, backend: TrainingBackend) -> Generation:
    parent_tables = None
    if any(member.state == "planned" for member in generation.members):
        try:
            parent_tables = backend.run_tables(state.spec.game, generation.parent)
        except (LookupError, ValueError):
            parent_tables = None  # the planned members fail, saying why
    members = tuple(advance_member(state, generation, member, backend, parent_tables) for member in generation.members)
    return dataclasses.replace(generation, members=members)


def _judged(state: SearchState, generation: Generation, accept: Accept) -> SearchState:
    members = generation.members
    best = winner([m.score for m in members], [m.state == "done" for m in members], state.higher_is_better)
    if best is None:
        error = f"no member of generation {generation.index} ended with its final evaluation, so none can carry the search on."
        return dataclasses.replace(state, status="failed", error=error)
    generation = dataclasses.replace(generation, winner=best)
    state = _with_last(state, generation)
    if generation.index + 1 >= state.spec.settings.generations:
        return dataclasses.replace(state, status="finished")
    champion = members[best]
    return _planned(state, champion.run, champion.checkpoint, champion.reward, accept)


def _planned(state: SearchState, parent: str, checkpoint: str | None, current: Mapping[str, Any], accept: Accept) -> SearchState:
    try:
        generation = plan_generation(state, parent, checkpoint, current, accept)
    except ValueError as error:
        return dataclasses.replace(state, status="failed", error=str(error))
    return dataclasses.replace(state, generations=(*state.generations, generation))


def _with_last(state: SearchState, generation: Generation) -> SearchState:
    return dataclasses.replace(state, generations=(*state.generations[:-1], generation))
