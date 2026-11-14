"""Pygame display of an ``"image"`` view payload (see ``views.image``). No game logic and no input handling."""

import io
from typing import Any

import pygame

from attentive_snake.views.image import png_bytes


class ImageWindow:
    """Shows image payloads ``scale`` times their size; the window is created on first render."""

    def __init__(self, width: int, height: int, scale: float = 1.0) -> None:
        self._size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self._screen: pygame.Surface | None = None

    def render(self, image: dict[str, Any], caption: str | None = None) -> None:
        screen = self._ensure_window()
        picture = pygame.image.load(io.BytesIO(png_bytes(image)), "frame.png")
        if picture.get_size() != self._size:
            picture = pygame.transform.smoothscale(picture, self._size)
        screen.blit(picture, (0, 0))
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
            self._screen = pygame.display.set_mode(self._size)
        return self._screen
