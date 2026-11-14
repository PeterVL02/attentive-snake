"""Watch a trained policy play its game in a pygame window.

    uv run python scripts/watch.py runs/snake/8x8 [--checkpoint PATH] [--attention [--layer -1]] [--sample]

Everything (game, rules, reward, model) is built from RUN_DIR/config.toml; the
latest checkpoint in RUN_DIR is used unless --checkpoint is given.
``--attention`` shades each cell by how much the class token attends to it. Models whose
tokens are whole frames (frame_attention) show the per-frame weights in the window title
instead; models with several frames of cells (temporal_transformer) show the newest frame
unless ``--frame`` picks another (0 = oldest, negative counts from the newest); a model
without attention (cnn) cannot be used with ``--attention``.
Esc or closing the window quits.
"""

import argparse
from pathlib import Path

import numpy as np
import pygame

from attentive_snake.training import latest_checkpoint
from attentive_snake.views.board_window import BoardWindow
from attentive_snake.watch import WatchFrame, WatchSession

# One pygame window per view kind (see attentive_snake.views); "image" has none yet.
WINDOWS = {"board": BoardWindow}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="default: the latest checkpoint in RUN_DIR")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cell-size", type=int, default=40)
    parser.add_argument("--sample", action="store_true", help="sample actions instead of taking the argmax")
    parser.add_argument("--attention", action="store_true", help="overlay the class-token attention heatmap")
    parser.add_argument("--layer", type=int, default=-1, help="transformer block whose attention is shown")
    parser.add_argument("--frame", type=int, default=-1, help="frame whose cells are shown (-1 = newest)")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = latest_checkpoint(args.run_dir)
        if args.checkpoint is None:
            parser.error(f"no checkpoint found in {args.run_dir}")
    return args


def make_session(args: argparse.Namespace) -> WatchSession:
    session = WatchSession(
        args.run_dir, args.checkpoint, sample=args.sample, attention=args.attention, layer=args.layer, frame=args.frame
    )
    if args.attention and session.info["attention_view"] is None:
        raise SystemExit(f"--attention: the {session.info['kind']} model has no attention to show")
    return session


def quit_requested() -> bool:
    return any(
        event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
        for event in pygame.event.get()
    )


def window_title(frame: WatchFrame) -> str:
    if frame.frame_weights is not None:
        return "frame attention (oldest..newest): " + " ".join(f"{w:.2f}" for w in frame.frame_weights)
    suffix = f"  ({frame.event})" if frame.done else ""
    return f"score {frame.score:g}, step {frame.steps}{suffix}"


def main() -> None:
    args = parse_args()
    try:
        session = make_session(args)
    except ValueError as error:  # e.g. --layer / --frame out of range
        raise SystemExit(f"--attention: {error}") from error
    print(f"loaded {session.info['checkpoint']} ({session.info['env_steps']} env steps)")

    window_type = WINDOWS.get(session.info["view_kind"])
    if window_type is None:
        raise SystemExit(f"no pygame window for the {session.info['view_kind']!r} view of {session.info['game']} yet")
    window = window_type(session.info["width"], session.info["height"], cell_size=args.cell_size)
    clock = pygame.time.Clock()
    frame = session.reset()
    while True:
        heatmap = None if frame.heatmap is None else np.array(frame.heatmap, dtype=np.float32)
        window.render(frame.view, heatmap, caption=window_title(frame))
        if quit_requested():  # after render: the window is created on the first render
            break
        frame = session.step()
        if frame.done:
            print(f"episode over: {frame.event}, score {frame.score:g}")
        clock.tick(args.fps)
    window.close()


if __name__ == "__main__":
    main()
