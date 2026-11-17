"""MinAtar (Young & Tian 2019, the ``minatar`` package) as one game: see ``game.GAME``.

One adapter for every title: the title is a rule (``[game] title``), listed in ``titles.TITLES``.
"""

from attentive_snake.games.minatar.game import GAME

__all__ = ["GAME"]
