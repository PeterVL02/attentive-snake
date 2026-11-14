"""Human play: keyboard -> SnakeGame -> PygameRenderer."""

import argparse

import pygame

from attentive_snake.games.snake.config import GameConfig
from attentive_snake.games.snake.engine import SnakeGame
from attentive_snake.games.snake.renderer import PygameRenderer
from attentive_snake.games.snake.types import Action

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


def play(config: GameConfig, fps: int = 8, cell_size: int = 30, seed: int | None = None) -> None:
    """Run an interactive game until the window is closed or Esc is pressed."""
    game = SnakeGame(config, seed=seed)
    renderer = PygameRenderer(config.width, config.height, cell_size)
    pygame.init()
    clock = pygame.time.Clock()
    try:
        _loop(game, renderer, clock, fps)
    finally:
        renderer.close()
        pygame.quit()


def _loop(game: SnakeGame, renderer: PygameRenderer, clock: pygame.time.Clock, fps: int) -> None:
    pending = game.state.direction
    renderer.render(game.state)
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
        renderer.render(game.state)
        clock.tick(fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play snake with the keyboard.")
    parser.add_argument("--width", type=int, default=GameConfig.width)
    parser.add_argument("--height", type=int, default=GameConfig.height)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=30)
    args = parser.parse_args()
    play(GameConfig(width=args.width, height=args.height), fps=args.fps, cell_size=args.cell_size)


if __name__ == "__main__":
    main()
