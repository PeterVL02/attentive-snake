"""Train a policy on a game with PPO: start a new run, resume one, or continue one under a new name (fork).

Every setting lives in a TOML experiment config (see configs/<game>/default.toml);
``[game] name`` says which game. Each run gets a directory (by convention
runs/<game>/<name>) with its config.toml, checkpoints, model.html, metrics.jsonl,
evaluations.jsonl (when [evaluation] interval > 0) and run.json (its start/resume/fork
history with descriptions).

    # new run (refuses a directory that already has checkpoints)
    uv run python scripts/train.py [--game snake] --run-dir runs/snake/8x8 [--config configs/snake/default.toml]
        [--width W --height H --num-envs N --seed S --steps STEPS --set section.key=value ...]

    # continue a run from its config.toml and latest checkpoint (the game comes from the config)
    uv run python scripts/train.py --resume runs/snake/8x8 [--steps 10000000]

    # a new run that starts from a checkpoint of another run of the same game, with changed settings
    uv run python scripts/train.py --run-dir runs/snake/8x8-lowlr --fork runs/snake/8x8
        [--fork-checkpoint checkpoint_0000500000.pt] [--set train.learning_rate=1e-4 ...]

All three take ``--description TEXT`` (saved in run.json),
``--keep-checkpoints / --no-keep-checkpoints`` and ``--eval-interval N --eval-games X
--eval-seed S`` (saved in config.toml; ``--eval-interval 0`` turns evaluation off).
``--steps`` sets ``train.total_env_steps``. When it extends a resumed run, the
linearly annealed learning rate is re-stretched over the new total. A forked
run counts its own steps from 0 with its own total and learning-rate schedule.
"""

import argparse
import dataclasses
import tomllib
from pathlib import Path

from attentive_snake.config import ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.experiment_io import game_name_of
from attentive_snake.config.layout import default_config_path
from attentive_snake.config.fork_rules import require_forkable
from attentive_snake.config.overrides import apply_overrides, parse_override
from attentive_snake.evaluation.hook import PeriodicEvaluation
from attentive_snake.factory import make_env, make_policy, make_vec_env
from attentive_snake.games.registry import DEFAULT_GAME, game_names, get_game
from attentive_snake.run_dir import (
    chosen_checkpoint,
    config_path,
    diagram_path,
    evaluations_path,
    game_runs_dir,
    list_checkpoints,
    metrics_path,
)
from attentive_snake.run_metadata import Lineage, RunEvent, append_event, clean_description
from attentive_snake.training import (
    ConsoleLogger,
    JsonlLogger,
    TeeLogger,
    Trainer,
    fork_from_checkpoint,
    latest_checkpoint,
    resume_from_checkpoint,
)
from attentive_snake.training.runtime import seed_everything
from attentive_snake.viz import diagram, save_diagram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_ROOT = Path("runs")
DEFAULT_RUN_NAME = "default"
CONFIG_CHANGES = ("game", "config", "width", "height", "num_envs", "seed", "set")  # not with --resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--run-dir", type=Path, help=f"directory of a new run (default: {RUNS_ROOT}/<game>/{DEFAULT_RUN_NAME})"
    )
    target.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue the run in RUN_DIR")
    parser.add_argument("--steps", type=int, help="set train.total_env_steps (also extends a resumed run)")
    parser.add_argument(
        "--keep-checkpoints",
        action=argparse.BooleanOptionalAction,
        help="keep every checkpoint (the default for new runs), or delete older ones after each save",
    )
    parser.add_argument("--description", help="note about this start/resume/fork, saved in the run's run.json")
    evaluation = parser.add_argument_group("evaluation while training (every mode; saved in config.toml)")
    evaluation.add_argument("--eval-interval", type=int, metavar="N", help="set evaluation.interval: evaluate every N env steps (0 = off)")
    evaluation.add_argument("--eval-games", type=int, metavar="X", help="set evaluation.games: games per evaluation")
    evaluation.add_argument("--eval-seed", type=int, metavar="S", help="set evaluation.seed: seed of the evaluation games")
    fork = parser.add_argument_group("continuing another run under a new name (with --run-dir)")
    fork.add_argument("--fork", type=Path, metavar="PARENT_RUN_DIR", help="start from a checkpoint of this run")
    fork.add_argument("--fork-checkpoint", type=Path, metavar="FILE", help="checkpoint of the parent (default: its latest)")
    new_run = parser.add_argument_group("new and forked runs only")
    new_run.add_argument(
        "--game",
        choices=game_names(),
        help=f"game of a new run (default: the --config file's [game] name, else {DEFAULT_GAME}); "
        "picks configs/<game>/default.toml and the default run directory",
    )
    new_run.add_argument(
        "--config", type=Path, help="experiment TOML (default: configs/<game>/default.toml; with --fork, the parent's config.toml)"
    )
    new_run.add_argument("--width", type=int, help="shortcut for --set game.width=W (grid games)")
    new_run.add_argument("--height", type=int, help="shortcut for --set game.height=H (grid games)")
    new_run.add_argument("--num-envs", type=int)
    new_run.add_argument("--seed", type=int)
    new_run.add_argument(
        "--set", action="append", metavar="SECTION.KEY=VALUE", help="change one setting, e.g. train.learning_rate=1e-4 (repeatable)"
    )
    return parser


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that do not fit the chosen mode (exits with a usage error)."""
    if args.resume is not None:
        given = [f"--{name.replace('_', '-')}" for name in (*CONFIG_CHANGES, "fork") if getattr(args, name) is not None]
        if given:
            parser.error(f"{', '.join(given)} cannot be used with --resume (the run's config.toml is used).")
        return
    if args.fork_checkpoint is not None and args.fork is None:
        parser.error("--fork-checkpoint needs --fork.")
    if args.fork is not None and args.run_dir is None:
        parser.error("--fork needs --run-dir: the directory of the new run.")
    args.run_dir = args.run_dir or game_runs_dir(RUNS_ROOT, requested_game(args)) / DEFAULT_RUN_NAME
    if args.fork is not None and args.run_dir.resolve() == args.fork.resolve():
        parser.error("--run-dir must differ from --fork; use --resume to continue the same run.")
    if latest_checkpoint(args.run_dir) is not None:
        parser.error(f"{args.run_dir} already contains checkpoints; choose another --run-dir, or --resume {args.run_dir}.")


def requested_game(args: argparse.Namespace) -> str:
    """The game of a new run: --game, else the --config file's [game] name, else the default game."""
    if args.game is not None:
        return args.game
    if args.config is not None:
        try:
            with args.config.open("rb") as file:
                return game_name_of(tomllib.load(file))
        except (OSError, tomllib.TOMLDecodeError, TypeError):
            pass  # reported properly when the config is loaded
    return DEFAULT_GAME


def replace_some[T](obj: T, **changes) -> T:
    """``dataclasses.replace`` with the ``None``-valued changes (options not given) left out."""
    return dataclasses.replace(obj, **{name: value for name, value in changes.items() if value is not None})


def with_run_options(config: ExperimentConfig, args: argparse.Namespace, run_dir: Path) -> ExperimentConfig:
    """``config`` with the options every mode accepts (--steps, --keep-checkpoints, --eval-*) and the run directory."""
    train = replace_some(
        config.train, total_env_steps=args.steps, keep_checkpoints=args.keep_checkpoints, checkpoint_dir=run_dir
    )
    evaluation = replace_some(config.evaluation, interval=args.eval_interval, games=args.eval_games, seed=args.eval_seed)
    return dataclasses.replace(config, train=train, evaluation=evaluation)


def with_config_changes(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """``config`` with the --set overrides, then the --width/--height/--num-envs/--seed shortcuts.

    ``ValueError`` if --game names another game than the config's.
    """
    if args.game is not None and args.game != config.game_name:
        raise ValueError(f"--game {args.game} does not match the config's game {config.game_name!r}.")
    shortcuts = [("game", key, value) for key, value in (("width", args.width), ("height", args.height)) if value is not None]
    config = apply_overrides(config, [parse_override(text) for text in args.set or ()] + shortcuts)
    return dataclasses.replace(
        config,
        env=replace_some(config.env, num_envs=args.num_envs),
        train=replace_some(config.train, seed=args.seed),
    )


def new_run_config(args: argparse.Namespace) -> ExperimentConfig:
    config_file = args.config or default_config_path(CONFIGS_DIR, requested_game(args))
    config = with_config_changes(load_experiment(config_file), args)
    return with_run_options(config, args, args.run_dir)


def resumed_run_config(args: argparse.Namespace) -> ExperimentConfig:
    # The run directory is wherever the run lives now, even if it was moved.
    return with_run_options(load_experiment(config_path(args.resume)), args, args.resume)


def forked_run_config(args: argparse.Namespace) -> ExperimentConfig:
    """The parent's config (or --config) with the changes applied; ``ValueError`` if a locked setting changed."""
    parent = load_experiment(config_path(args.fork))
    config = with_config_changes(load_experiment(args.config) if args.config else parent, args)
    config = with_run_options(config, args, args.run_dir)
    require_forkable(parent, config)
    return config


def fork_checkpoint(parent_dir: Path, requested: Path | None) -> Path:
    """The parent's checkpoint to start from: ``requested`` (a file name or a path inside it), else its latest."""
    if not list_checkpoints(parent_dir):
        raise ValueError(f"{parent_dir} has no checkpoints to continue from.")
    return chosen_checkpoint(parent_dir, requested)


def evaluation_hooks(config: ExperimentConfig, run_dir: Path) -> list[PeriodicEvaluation]:
    """The periodic evaluation of the run, if ``[evaluation] interval`` is set; it appends to the run's evaluations.jsonl."""
    if not config.evaluation.enabled:
        return []
    game = get_game(config.game_name)
    return [PeriodicEvaluation(game, lambda: make_env(config), config.evaluation, evaluations_path(run_dir))]


def resume(trainer: Trainer, run_dir: Path, description: str | None) -> RunEvent:
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        print(f"no checkpoint in {run_dir}; starting from step 0")
        return RunEvent("resume", 0, description)
    steps = resume_from_checkpoint(trainer, checkpoint)
    print(f"resumed from {checkpoint} at {steps} env steps (target {trainer.config.total_env_steps})")
    return RunEvent("resume", steps, description)


def fork(trainer: Trainer, parent_dir: Path, checkpoint: Path, description: str | None) -> RunEvent:
    parent_steps = fork_from_checkpoint(trainer, checkpoint)
    print(f"forked from {checkpoint} ({parent_steps} env steps of {parent_dir.name}); this run counts from 0")
    return RunEvent("fork", 0, description, Lineage(parent_dir.resolve().name, checkpoint.name, parent_steps))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    check_args(parser, args)
    run_dir = args.resume or args.run_dir
    description = clean_description(args.description)
    try:
        if args.resume:
            config = resumed_run_config(args)
        elif args.fork:
            config = forked_run_config(args)
            source = fork_checkpoint(args.fork, args.fork_checkpoint)
        else:
            config = new_run_config(args)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    save_experiment(config, config_path(run_dir))

    env = make_vec_env(config)
    seed_everything(config.train.seed)
    policy = make_policy(config, env)
    print(diagram(policy))
    save_diagram(policy, diagram_path(run_dir))

    logger = TeeLogger(ConsoleLogger(), JsonlLogger(metrics_path(run_dir)))
    trainer = Trainer(policy, env, config.train, logger, hooks=evaluation_hooks(config, run_dir))
    if args.resume:
        event = resume(trainer, run_dir, description)
    elif args.fork:
        event = fork(trainer, args.fork, source, description)
    else:
        event = RunEvent("start", 0, description)
    append_event(run_dir, event)
    if not trainer.train():
        print(f"nothing to do: all {trainer.num_updates} updates of total_env_steps={config.train.total_env_steps} are done")


if __name__ == "__main__":
    main()
