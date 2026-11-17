"""Planning a generation: its parent, and one member per proposed reward table, named and not yet queued."""

from collections.abc import Callable, Mapping
from typing import Any

from attentive_snake.reward_search.naming import member_run
from attentive_snake.reward_search.perturbation import population_rewards
from attentive_snake.reward_search.state import Generation, Member, SearchState


def plan_generation(
    state: SearchState, parent: str, checkpoint: str | None, current: Mapping[str, Any], accept: Callable[[Mapping[str, Any]], bool]
) -> Generation:
    """The next generation of ``state``: members forking ``parent``'s ``checkpoint``, member 0 on ``current``.

    ``ValueError`` as ``population_rewards``.
    """
    spec = state.spec
    index = len(state.generations)
    rewards = population_rewards(current, state.start_reward, spec.fields, spec.settings, index, accept)
    members = tuple(Member(m, member_run(spec.name, index, m), reward) for m, reward in enumerate(rewards))
    return Generation(index, parent, checkpoint, members)
