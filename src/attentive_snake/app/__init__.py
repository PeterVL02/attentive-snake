"""Localhost web app for training, resuming and watching runs: ``python -m attentive_snake.app``."""

from attentive_snake.app.application import create_app

__all__ = ["create_app"]
