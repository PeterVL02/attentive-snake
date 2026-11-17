"""The experiment tables of a member run: its parent's, with the member's reward, its training length and a final evaluation."""

import copy
from collections.abc import Mapping
from typing import Any

from attentive_snake.reward_search.settings import SearchSettings


def member_tables(parent: Mapping[str, Mapping[str, Any]], reward: Mapping[str, Any], settings: SearchSettings) -> dict[str, dict[str, Any]]:
    """``parent``'s tables with ``[reward]`` replaced, ``steps_per_member`` env steps to train (a fork counts from 0)
    and one evaluation of ``evaluation_games`` games at the end (the interval equals the length; the seed stays the parent's,
    so every member plays the same games)."""
    tables = {section: dict(values) for section, values in copy.deepcopy(dict(parent)).items()}
    tables["reward"] = dict(reward)
    tables.setdefault("train", {})["total_env_steps"] = settings.steps_per_member
    evaluation = tables.setdefault("evaluation", {})
    evaluation["interval"] = settings.steps_per_member
    evaluation["games"] = settings.evaluation_games
    return tables
