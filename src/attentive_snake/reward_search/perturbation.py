"""Proposing the reward tables of one generation: the current table unchanged, then seeded, bounded changes of it.

The rule for one selected coefficient, with ``s`` its value at the start of the search and
``size = |s|`` (``zero_scale`` when ``s`` is 0, which has no size of its own):

* move: the current value plus a uniform draw from ``[-perturbation * size, +perturbation * size]``;
* bound: never farther than ``max_change * size`` from ``s``;
* sign: unless ``allow_sign_change``, a coefficient that started positive stays ``>= 0`` and one that
  started negative ``<= 0``; one that started at 0 may go either way.

A table the game rejects (e.g. a scale that must be positive came out 0) is drawn again.
Values are rounded to 4 significant digits, so tables read well in the app and in config.toml.
"""

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from attentive_snake.reward_search.settings import SearchSettings

MAX_DRAWS = 100
SIGNIFICANT_DIGITS = 4


def _rounded(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def generation_rng(seed: int, generation: int) -> random.Random:
    """The random stream of one generation: the same for the same seed and generation, whatever happened before."""
    return random.Random(f"reward-search/{seed}/{generation}")


def perturbed_value(value: float, start: float, settings: SearchSettings, rng: random.Random) -> float:
    """One move of a coefficient now at ``value`` that started the search at ``start`` (see the module docstring)."""
    size = abs(start) if start != 0 else settings.zero_scale
    proposal = value + rng.uniform(-1.0, 1.0) * settings.perturbation * size
    reach = settings.max_change * size
    proposal = min(max(proposal, start - reach), start + reach)
    if not settings.allow_sign_change:
        proposal = max(proposal, 0.0) if start > 0 else min(proposal, 0.0) if start < 0 else proposal
    return _rounded(proposal)


def population_rewards(
    current: Mapping[str, Any],
    start: Mapping[str, Any],
    fields: Sequence[str],
    settings: SearchSettings,
    generation: int,
    accept: Callable[[Mapping[str, Any]], bool],
) -> list[dict[str, Any]]:
    """``settings.population`` tables: ``current`` unchanged first, then changed copies (only ``fields`` change).

    ``accept(table)`` says whether the game takes a table; ``ValueError`` if no acceptable change is
    found in ``MAX_DRAWS`` draws for one member.
    """
    rng = generation_rng(settings.seed, generation)
    tables = [dict(current)]
    for _ in range(settings.population - 1):
        for _ in range(MAX_DRAWS):
            table = dict(current) | {name: perturbed_value(float(current[name]), float(start[name]), settings, rng) for name in fields}
            if accept(table):
                break
        else:
            raise ValueError(f"no changed reward table the game accepts was found in {MAX_DRAWS} draws.")
        tables.append(table)
    return tables
