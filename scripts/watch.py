"""Watch a trained policy play its game in a pygame window.

    uv run python scripts/watch.py runs/snake/8x8 [--checkpoint PATH] [--attention [--layer -1] [--frame -1]] [--sample]
    uv run python scripts/watch.py runs/lunar_lander/NAME [--attention] [--scale 1.0]

Everything (game, rules, reward, model) is built from RUN_DIR/config.toml; the
latest checkpoint in RUN_DIR is used unless --checkpoint is given.
Board games (Snake) are drawn cell by cell; ``--attention`` shades each cell by how much
the class token attends to it. Image games (Lunar Lander) show the game's own picture;
``--attention`` puts the class token's weight on each input number (vector observations)
in the window title. Models whose tokens are whole frames (frame_attention) show the
per-frame weights in the window title instead; models with several frames of tokens
(temporal_transformer) show the newest frame unless ``--frame`` picks another (0 = oldest,
negative counts from the newest); a model without attention (cnn, mlp) cannot be used
with ``--attention``. Esc or closing the window quits.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pygame

from attentive_snake.training import latest_checkpoint
from attentive_snake.views.board_window import BoardWindow
from attentive_snake.views.image_window import ImageWindow
from attentive_snake.watch import WatchFrame, WatchSession

type Draw = Callable[[WatchFrame, str], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="default: the latest checkpoint in RUN_DIR")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cell-size", type=int, default=40, help="board games: pixels per cell")
    parser.add_argument("--scale", type=float, default=1.0, help="image games: size of the window relative to the picture")
    parser.add_argument("--sample", action="store_true", help="sample actions instead of taking the argmax")
    parser.add_argument("--attention", action="store_true", help="show the class-token attention")
    parser.add_argument("--layer", type=int, default=-1, help="transformer block whose attention is shown")
    parser.add_argument("--frame", type=int, default=-1, help="history frame whose tokens are shown (-1 = newest)")
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


def board_display(info: dict, args: argparse.Namespace) -> tuple[BoardWindow, Draw]:
    window = BoardWindow(info["width"], info["height"], cell_size=args.cell_size)

    def draw(frame: WatchFrame, caption: str) -> None:
        heatmap = None if frame.heatmap is None else np.array(frame.heatmap, dtype=np.float32)
        window.render(frame.view, heatmap, caption=caption)

    return window, draw


def image_display(info: dict, args: argparse.Namespace) -> tuple[ImageWindow, Draw]:
    window = ImageWindow(info["width"], info["height"], scale=args.scale)
    return window, lambda frame, caption: window.render(frame.view, caption=caption)


# One pygame display per view kind (see attentive_snake.views).
DISPLAYS = {"board": board_display, "image": image_display}


def quit_requested() -> bool:
    return any(
        event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
        for event in pygame.event.get()
    )


def window_title(frame: WatchFrame, element_names: list[str] | None) -> str:
    if frame.frame_weights is not None:
        return "frame attention (oldest..newest): " + " ".join(f"{w:.2f}" for w in frame.frame_weights)
    if frame.element_weights is not None and element_names is not None:
        return "attention: " + ", ".join(f"{name} {w:.2f}" for name, w in zip(element_names, frame.element_weights))
    suffix = f"  ({frame.event})" if frame.done else ""
    return f"score {frame.score:g}, step {frame.steps}{suffix}"


def main() -> None:
    args = parse_args()
    try:
        session = make_session(args)
    except ValueError as error:  # e.g. --layer / --frame out of range
        raise SystemExit(f"--attention: {error}") from error
    print(f"loaded {session.info['checkpoint']} ({session.info['env_steps']} env steps)")

    display = DISPLAYS.get(session.info["view_kind"])
    if display is None:
        raise SystemExit(f"no pygame window for the {session.info['view_kind']!r} view of {session.info['game']}")
    window, draw = display(session.info, args)
    clock = pygame.time.Clock()
    frame = session.reset()
    while True:
        draw(frame, window_title(frame, session.info["element_names"]))
        if quit_requested():  # after drawing: the window is created on the first render
            break
        frame = session.step()
        if frame.done:
            print(f"episode over: {frame.event}, score {frame.score:g}")
        clock.tick(args.fps)
    window.close()


if __name__ == "__main__":
    main()
