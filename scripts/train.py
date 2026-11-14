"""Train a policy on Snake with PPO: start a new run, resume one, or continue one under a new name (fork).

Every setting lives in a TOML experiment config (see configs/default.toml).
Each run gets a directory with its config.toml, checkpoints, model.html,
metrics.jsonl and run.json (its start/resume/fork history with descriptions).

    # new run (refuses a directory that already has checkpoints)
    uv run python scripts/train.py --run-dir runs/8x8 [--config configs/default.toml]
        [--width W --height H --num-envs N --seed S --steps STEPS --set section.key=value ...]

    # continue a run from its config.toml and latest checkpoint
    uv run python scripts/train.py --resume runs/8x8 [--steps 10000000]

    # a new run that starts from a checkpoint of another run, with changed settings
    uv run python scripts/train.py --run-dir runs/8x8-lowlr --fork runs/8x8
        [--fork-checkpoint checkpoint_0000500000.pt] [--set train.learning_rate=1e-4 ...]

All three take ``--description TEXT`` (saved in run.json) and
``--keep-checkpoints / --no-keep-checkpoints`` (saved in config.toml).
``--steps`` sets ``train.total_env_steps``. When it extends a resumed run, the
linearly annealed learning rate is re-stretched over the new total. A forked
run counts its own steps from 0 with its own total and learning-rate schedule.
"""

import argparse
import dataclasses
from pathlib import Path

from attentive_snake.config import ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.fork_rules import require_forkable
from attentive_snake.config.overrides import apply_overrides, parse_override
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.run_dir import config_path, diagram_path, list_checkpoints, metrics_path
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

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
DEFAULT_RUN_DIR = Path("runs/default")
CONFIG_CHANGES = ("config", "width", "height", "num_envs", "seed", "set")  # not with --resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-dir", type=Path, help=f"directory of a new run (default: {DEFAULT_RUN_DIR})")
    target.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue the run in RUN_DIR")
    parser.add_argument("--steps", type=int, help="set train.total_env_steps (also extends a resumed run)")
    parser.add_argument(
        "--keep-checkpoints",
        action=argparse.BooleanOptionalAction,
        help="keep every checkpoint (the default for new runs), or delete older ones after each save",
    )
    parser.add_argument("--description", help="note about this start/resume/fork, saved in the run's run.json")
    fork = parser.add_argument_group("continuing another run under a new name (with --run-dir)")
    fork.add_argument("--fork", type=Path, metavar="PARENT_RUN_DIR", help="start from a checkpoint of this run")
    fork.add_argument("--fork-checkpoint", type=Path, metavar="FILE", help="checkpoint of the parent (default: its latest)")
    new_run = parser.add_argument_group("new and forked runs only")
    new_run.add_argument(
        "--config", type=Path, help=f"experiment TOML (default: {DEFAULT_CONFIG}; with --fork, the parent's config.toml)"
    )
    new_run.add_argument("--width", type=int)
    new_run.add_argument("--height", type=int)
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
    args.run_dir = args.run_dir or DEFAULT_RUN_DIR
    if args.fork is not None and args.run_dir.resolve() == args.fork.resolve():
        parser.error("--run-dir must differ from --fork; use --resume to continue the same run.")
    if latest_checkpoint(args.run_dir) is not None:
        parser.error(f"{args.run_dir} already contains checkpoints; choose another --run-dir, or --resume {args.run_dir}.")


def replace_some[T](obj: T, **changes) -> T:
    """``dataclasses.replace`` with the ``None``-valued changes (options not given) left out."""
    return dataclasses.replace(obj, **{name: value for name, value in changes.items() if value is not None})


def with_run_options(config: ExperimentConfig, args: argparse.Namespace, run_dir: Path) -> ExperimentConfig:
    """``config`` with the options every mode accepts (--steps, --keep-checkpoints) and the run directory."""
    train = replace_some(
        config.train, total_env_steps=args.steps, keep_checkpoints=args.keep_checkpoints, checkpoint_dir=run_dir
    )
    return dataclasses.replace(config, train=train)


def with_config_changes(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """``config`` with the --set overrides, then the --width/--height/--num-envs/--seed shortcuts."""
    config = apply_overrides(config, [parse_override(text) for text in args.set or ()])
    return dataclasses.replace(
        config,
        game=replace_some(config.game, width=args.width, height=args.height),
        env=replace_some(config.env, num_envs=args.num_envs),
        train=replace_some(config.train, seed=args.seed),
    )


def new_run_config(args: argparse.Namespace) -> ExperimentConfig:
    config = with_config_changes(load_experiment(args.config or DEFAULT_CONFIG), args)
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
    checkpoints = list_checkpoints(parent_dir)
    if not checkpoints:
        raise ValueError(f"{parent_dir} has no checkpoints to continue from.")
    if requested is None:
        return checkpoints[-1]
    for path in checkpoints:
        if path.name == requested.name and (requested.parent == Path() or requested.resolve() == path.resolve()):
            return path
    raise ValueError(f"{requested} is not a checkpoint of {parent_dir}; it has {', '.join(p.name for p in checkpoints)}.")


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

    trainer = Trainer(policy, env, config.train, TeeLogger(ConsoleLogger(), JsonlLogger(metrics_path(run_dir))))
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
