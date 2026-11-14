"""``PeriodicEvaluation``: the ``Trainer`` hook that evaluates the policy every ``[evaluation] interval`` env steps."""

from collections.abc import Callable
from pathlib import Path

from attentive_snake.evaluation.config import EvalConfig
from attentive_snake.evaluation.evaluate import evaluate_policy
from attentive_snake.evaluation.log import append_evaluation
from attentive_snake.evaluation.player import EnvFactory
from attentive_snake.evaluation.record import Evaluation
from attentive_snake.evaluation.rng import preserved_rng_state
from attentive_snake.evaluation.schedule import evaluation_due
from attentive_snake.evaluation.settings import settings_from_config
from attentive_snake.games.spec import GameSpec
from attentive_snake.training.hooks import UpdateEnd


def summary_line(evaluation: Evaluation) -> str:
    """``evaluation steps=... games=... mean_score=...`` for the training log."""
    shown = {key: value for key, value in evaluation.statistics.items() if key != "games"}
    values = " ".join(f"{key}={'N/A' if value is None else f'{value:.4g}'}" for key, value in shown.items())
    return f"evaluation steps={evaluation.env_steps} games={evaluation.settings.games} {values}"


class PeriodicEvaluation:
    """When an update is due (``schedule.evaluation_due``): make sure a checkpoint exists at
    that step, play ``config.games`` greedy games on fresh environments from ``make_env``,
    and append the result (``source="training"``) to ``log_path``.

    The global random streams are restored afterwards (``rng.preserved_rng_state``) and the
    training environments are never touched, so training continues exactly as it would
    have without evaluating.
    """

    def __init__(
        self,
        game: GameSpec,
        make_env: EnvFactory,
        config: EvalConfig,
        log_path: Path,
        announce: Callable[[str], None] = print,
    ) -> None:
        self._game = game
        self._make_env = make_env
        self._config = config
        self._log_path = Path(log_path)
        self._announce = announce

    def __call__(self, update: UpdateEnd) -> None:
        if not evaluation_due(update.previous_env_steps, update.env_steps, self._config.interval, update.is_last):
            return
        checkpoint = update.save_checkpoint()
        settings = settings_from_config(self._config, str(update.device))
        with preserved_rng_state():
            evaluation = evaluate_policy(
                update.policy,
                self._game,
                self._make_env,
                settings,
                source="training",
                env_steps=update.env_steps,
                checkpoint=None if checkpoint is None else checkpoint.name,
            )
        append_evaluation(self._log_path, evaluation)
        self._announce(summary_line(evaluation))
