"""Evaluate a checkpoint of a run: play complete games with the most likely action at every step and print the statistics.

    uv run python scripts/evaluate.py runs/snake/8x8 [--checkpoint FILE] [--games X] [--seed S]
        [--max-steps N] [--save]

The defaults are the run's ``[evaluation]`` settings (config.toml) and its latest checkpoint.
With the settings of an evaluation recorded while training (same checkpoint, games, seed and
step cap, on the same device) the numbers are exactly the recorded ones; the script says
whether they match. ``--save`` appends the result to the run's evaluations.jsonl, marked
``"source": "manual"``.
"""

import argparse
import sys
from pathlib import Path

from attentive_snake.config import load_experiment
from attentive_snake.evaluation.checkpoint_evaluation import evaluate_checkpoint
from attentive_snake.evaluation.log import append_evaluation, read_evaluations
from attentive_snake.evaluation.matching import Reproduction, find_reproduction
from attentive_snake.evaluation.settings import settings_from_config
from attentive_snake.evaluation.statistics import metric_definitions
from attentive_snake.evaluation.table import evaluation_table
from attentive_snake.games.registry import get_game
from attentive_snake.run_dir import chosen_checkpoint, config_path, evaluations_path, is_run_dir
from attentive_snake.training.runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, metavar="RUN_DIR", help="the run, e.g. runs/snake/8x8")
    parser.add_argument("--checkpoint", type=Path, metavar="FILE", help="checkpoint file name of the run (or its path; default: the latest)")
    parser.add_argument("--games", type=int, metavar="X", help="number of games (default: the run's evaluation.games)")
    parser.add_argument("--seed", type=int, metavar="S", help="seed of the games (default: the run's evaluation.seed)")
    parser.add_argument("--max-steps", type=int, metavar="N", help="per-game step cap (default: the run's evaluation.max_episode_steps)")
    parser.add_argument("--save", action="store_true", help="append the result to the run's evaluations.jsonl (as a manual evaluation)")
    return parser


def reproduction_line(reproduction: Reproduction | None) -> str:
    if reproduction is None:
        return "No recorded evaluation of this checkpoint with these settings to compare with."
    when = f"the {reproduction.recorded.source} evaluation of {reproduction.recorded.timestamp}"
    if reproduction.identical:
        return f"Identical to {when}."
    device_note = "" if reproduction.same_device else f" (that one ran on {reproduction.recorded.settings.device})"
    return f"DIFFERENT from {when}{device_note}."


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not is_run_dir(args.run_dir):
        parser.error(f"{args.run_dir} is not a run directory (no config.toml).")
    try:
        config = load_experiment(config_path(args.run_dir))
        checkpoint = chosen_checkpoint(args.run_dir, args.checkpoint)
        device = str(resolve_device(config.train.device))
        settings = settings_from_config(config.evaluation, device, games=args.games, seed=args.seed, max_episode_steps=args.max_steps)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))

    print(f"evaluating {checkpoint} ...", file=sys.stderr, flush=True)
    evaluation = evaluate_checkpoint(args.run_dir, checkpoint, settings)
    print(evaluation_table(evaluation, metric_definitions(get_game(config.game_name))))
    print(reproduction_line(find_reproduction(read_evaluations(evaluations_path(args.run_dir)), evaluation)))
    if args.save:
        append_evaluation(evaluations_path(args.run_dir), evaluation)
        print(f"saved to {evaluations_path(args.run_dir)}")


if __name__ == "__main__":
    main()
