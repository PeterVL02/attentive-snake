"""Choosing the winner of a generation from its members' objective values."""

from collections.abc import Sequence

UNCHANGED = 0  # the member that keeps the current reward table


def winner(scores: Sequence[float | None], finished: Sequence[bool], higher_is_better: bool) -> int | None:
    """The index of the winning member, or ``None`` if no member can carry the search on.

    ``scores[i]`` is member ``i``'s objective value (``None``: not available), ``finished[i]``
    whether it ended with its final evaluation (only those can win). The best score wins; on a
    tie the unchanged member (index 0), else the lowest index. If no finished member has a score,
    the unchanged member carries on if it finished.
    """
    candidates = [i for i, done in enumerate(finished) if done and scores[i] is not None]
    if not candidates:
        return UNCHANGED if finished and finished[UNCHANGED] else None
    sign = 1.0 if higher_is_better else -1.0
    return max(candidates, key=lambda i: (sign * scores[i], i == UNCHANGED, -i))
