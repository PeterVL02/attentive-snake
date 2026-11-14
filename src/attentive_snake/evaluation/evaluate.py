"""Evaluating a policy that is already loaded: play the games, compute the statistics, wrap them in an ``Evaluation``."""

import time

from torch import nn

from attentive_snake.evaluation.generic_statistics import outcome_counts
from attentive_snake.evaluation.player import EnvFactory, Progress, play_games
from attentive_snake.evaluation.record import Evaluation, Source
from attentive_snake.evaluation.settings import EvalSettings
from attentive_snake.evaluation.statistics import evaluation_statistics
from attentive_snake.games.spec import GameSpec


def evaluate_policy(
    policy: nn.Module,
    game: GameSpec,
    make_env: EnvFactory,
    settings: EvalSettings,
    *,
    source: Source,
    env_steps: int,
    checkpoint: str | None,
    progress: Progress | None = None,
) -> Evaluation:
    """Play ``settings.games`` greedy games of ``game`` with ``policy`` (on ``settings.device``) and summarise them."""
    start = time.perf_counter()
    records = play_games(policy, make_env, settings, progress)
    return Evaluation(
        source=source,
        env_steps=env_steps,
        checkpoint=checkpoint,
        settings=settings,
        statistics=evaluation_statistics(game, records),
        outcomes=outcome_counts(records),
        seconds=round(time.perf_counter() - start, 3),
    )
