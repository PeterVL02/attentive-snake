"""Headless watching of a trained policy: a session that yields JSON-friendly frames."""

from attentive_snake.watch.frame import WatchFrame
from attentive_snake.watch.session import WatchSession

__all__ = ["WatchFrame", "WatchSession"]
