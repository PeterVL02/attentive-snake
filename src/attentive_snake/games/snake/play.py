"""Human play: keyboard -> SnakeGame -> BoardWindow.

    uv run python -m attentive_snake.games.snake.play [--width 10 --height 10 --fps 8 --cell-size 30]
"""

import argparse

import pygame

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.engine import SnakeGame
from attentive_snake.games.snake.types import Action, GameState
from attentive_snake.games.snake.view import board_view
from attentive_snake.views.board_window import BoardWindow

_KEY_TO_ACTION: dict[int, Action] = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
}


def key_to_action(key: int) -> Action | None:
    """Map a pygame key code to an Action (arrows and WASD), else None."""
    return _KEY_TO_ACTION.get(key)


def play(config: SnakeConfig, fps: int = 8, cell_size: int = 30, seed: int | None = None) -> None:
    """Run an interactive game until the window is closed or Esc is pressed."""
    game = SnakeGame(config, seed=seed)
    renderer = BoardWindow(config.width, config.height, cell_size)
    pygame.init()
    clock = pygame.time.Clock()
    try:
        _loop(game, renderer, clock, fps)
    finally:
        renderer.close()
        pygame.quit()


def _loop(game: SnakeGame, renderer: BoardWindow, clock: pygame.time.Clock, fps: int) -> None:
    pending = game.state.direction
    _render(renderer, game.state)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return
            if event.key == pygame.K_r and game.state.done:
                game.reset()
                pending = game.state.direction
                continue
            action = key_to_action(event.key)
            if action is not None:
                pending = action
        if not game.state.done:
            game.step(pending)
        _render(renderer, game.state)
        clock.tick(fps)


def _render(renderer: BoardWindow, state: GameState) -> None:
    suffix = "  (game over)" if state.done else ""
    renderer.render(board_view(state), caption=f"Snake - score {state.score}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play snake with the keyboard.")
    parser.add_argument("--width", type=int, default=SnakeConfig.width)
    parser.add_argument("--height", type=int, default=SnakeConfig.height)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=30)
    args = parser.parse_args()
    play(SnakeConfig(width=args.width, height=args.height), fps=args.fps, cell_size=args.cell_size)


if __name__ == "__main__":
    main()
