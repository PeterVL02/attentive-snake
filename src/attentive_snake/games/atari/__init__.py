"""Atari 2600 games (Gymnasium's ``ALE/<Title>-v5`` through ``ale-py``) as one game: see ``game.GAME``.

One adapter for every title: the title is a rule (``[game] title``), listed in ``titles.TITLES``.
"""

from attentive_snake.games.atari.game import GAME

__all__ = ["GAME"]
