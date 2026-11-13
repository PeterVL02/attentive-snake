"""Train a transformer policy on Snake with PPO.

Every setting lives in a TOML experiment config (see configs/default.toml).
Each run gets a directory with its config.toml, checkpoints, model.html and metrics.jsonl.

    # new run (refuses a directory that already has checkpoints)
    uv run python scripts/train.py --run-dir runs/8x8 [--config configs/default.toml]
        [--width W --height H --num-envs N --seed S --steps STEPS]

    # continue a run from its config.toml and latest checkpoint
    uv run python scripts/train.py --resume runs/8x8 [--steps 10000000]

``--steps`` sets ``train.total_env_steps`` and is saved to the run's config.toml.
When it extends a run, the linearly annealed learning rate is re-stretched over
the new total (it is what it would have been had the run been planned that long).
"""

import argparse
import dataclasses
from pathlib import Path

from attentive_snake.config import ExperimentConfig, load_experiment, save_experiment
from attentive_snake.factory import make_policy, make_vec_env
from attentive_snake.run_dir import config_path, diagram_path, metrics_path
from attentive_snake.training import (
    ConsoleLogger,
    JsonlLogger,
    TeeLogger,
    Trainer,
    latest_checkpoint,
    resume_from_checkpoint,
)
from attentive_snake.training.runtime import seed_everything
from attentive_snake.viz import diagram, save_diagram

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
NEW_RUN_ONLY = ("config", "width", "height", "num_envs", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-dir", type=Path, default=Path("runs/default"), help="directory of a new run")
    target.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue the run in RUN_DIR")
    parser.add_argument("--steps", type=int, help="set train.total_env_steps (also extends a resumed run)")
    new_run = parser.add_argument_group("new runs only")
    new_run.add_argument("--config", type=Path, help=f"experiment TOML (default: {DEFAULT_CONFIG})")
    new_run.add_argument("--width", type=int)
    new_run.add_argument("--height", type=int)
    new_run.add_argument("--num-envs", type=int)
    new_run.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.resume is not None:
        given = [f"--{name.replace('_', '-')}" for name in NEW_RUN_ONLY if getattr(args, name) is not None]
        if given:
            parser.error(f"{', '.join(given)} cannot be used with --resume (the run's config.toml is used).")
    elif latest_checkpoint(args.run_dir) is not None:
        parser.error(f"{args.run_dir} already contains checkpoints; use --resume {args.run_dir} to continue it.")
    return args


def replace_some[T](obj: T, **changes) -> T:
    """``dataclasses.replace`` with the ``None``-valued changes (options not given) left out."""
    return dataclasses.replace(obj, **{name: value for name, value in changes.items() if value is not None})


def new_run_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_experiment(args.config or DEFAULT_CONFIG)
    return dataclasses.replace(
        config,
        game=replace_some(config.game, width=args.width, height=args.height),
        env=replace_some(config.env, num_envs=args.num_envs),
        train=replace_some(config.train, seed=args.seed, total_env_steps=args.steps, checkpoint_dir=args.run_dir),
    )


def resumed_run_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_experiment(config_path(args.resume))
    # The run directory is wherever the run lives now, even if it was moved.
    return dataclasses.replace(
        config, train=replace_some(config.train, total_env_steps=args.steps, checkpoint_dir=args.resume)
    )


def main() -> None:
    args = parse_args()
    run_dir = args.resume or args.run_dir
    config = resumed_run_config(args) if args.resume else new_run_config(args)
    checkpoint = latest_checkpoint(run_dir) if args.resume else None
    save_experiment(config, config_path(run_dir))

    env = make_vec_env(config)
    seed_everything(config.train.seed)
    policy = make_policy(config, env)
    print(diagram(policy))
    save_diagram(policy, diagram_path(run_dir))

    trainer = Trainer(policy, env, config.train, TeeLogger(ConsoleLogger(), JsonlLogger(metrics_path(run_dir))))
    if checkpoint is not None:
        steps = resume_from_checkpoint(trainer, checkpoint)
        print(f"resumed from {checkpoint} at {steps} env steps (target {config.train.total_env_steps})")
    elif args.resume:
        print(f"no checkpoint in {run_dir}; starting from step 0")
    if not trainer.train():
        print(f"nothing to do: all {trainer.num_updates} updates of total_env_steps={config.train.total_env_steps} are done")


if __name__ == "__main__":
    main()
