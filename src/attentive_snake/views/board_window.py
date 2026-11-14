"""Pygame drawing of a ``"board"`` view payload (see ``views.board``). No game logic and no input handling."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

Color = tuple[int, int, int]


@dataclass(frozen=True)
class RenderConfig:
    background: Color = (20, 20, 24)
    grid_line: Color = (32, 32, 38)
    snake_body: Color = (60, 180, 75)
    snake_head: Color = (120, 230, 120)
    food: Color = (220, 60, 60)
    cell_padding: int = 1
    heatmap: Color = (255, 200, 0)
    heatmap_max_alpha: int = 170


class BoardWindow:
    """Draws board payloads in a window; the window is created on first render."""

    def __init__(self, width: int, height: int, cell_size: int = 30, config: RenderConfig = RenderConfig()) -> None:
        self._width = width
        self._height = height
        self._cell_size = cell_size
        self._config = config
        self._screen: pygame.Surface | None = None

    def render(self, board: dict[str, Any], heatmap: np.ndarray | None = None, caption: str | None = None) -> None:
        """Draw ``board``; an optional (height, width) heatmap in [0, 1] tints the cells."""
        screen = self._ensure_window()
        screen.fill(self._config.background)
        if heatmap is not None:
            self._draw_heatmap(screen, heatmap)
        self._draw_grid(screen)
        if board["food"] is not None:
            self._draw_cell(screen, board["food"], self._config.food)
        body = board["snake"]
        for cell in body[1:]:
            self._draw_cell(screen, cell, self._config.snake_body)
        if body:
            self._draw_cell(screen, body[0], self._config.snake_head)
        if caption is not None:
            pygame.display.set_caption(caption)
        pygame.display.flip()

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None

    def _ensure_window(self) -> pygame.Surface:
        if self._screen is None:
            pygame.display.init()
            size = (self._width * self._cell_size, self._height * self._cell_size)
            self._screen = pygame.display.set_mode(size)
        return self._screen

    def _draw_grid(self, screen: pygame.Surface) -> None:
        size = self._cell_size
        for x in range(1, self._width):
            pygame.draw.line(screen, self._config.grid_line, (x * size, 0), (x * size, self._height * size))
        for y in range(1, self._height):
            pygame.draw.line(screen, self._config.grid_line, (0, y * size), (self._width * size, y * size))

    def _draw_heatmap(self, screen: pygame.Surface, heatmap: np.ndarray) -> None:
        if heatmap.shape != (self._height, self._width):
            raise ValueError(
                f"heatmap shape {heatmap.shape} does not match grid {(self._height, self._width)}"
            )
        size = self._cell_size
        overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        for y in range(self._height):
            for x in range(self._width):
                value = float(np.clip(heatmap[y, x], 0.0, 1.0))
                overlay.fill((*self._config.heatmap, round(value * self._config.heatmap_max_alpha)))
                screen.blit(overlay, (x * size, y * size))

    def _draw_cell(self, screen: pygame.Surface, cell: Any, color: Color) -> None:
        pad = self._config.cell_padding
        size = self._cell_size
        rect = pygame.Rect(cell[0] * size + pad, cell[1] * size + pad, size - 2 * pad, size - 2 * pad)
        pygame.draw.rect(screen, color, rect)
