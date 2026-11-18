"""Two-player Atari 2600 games (the Arcade Learning Environment's multi-player modes) as one game: see ``game.GAME``.

One policy learns by playing both players against itself (self-play, ``match.VersusMatch``);
evaluation and watching put it against a seeded random player (``env.VersusEnv``). The title is
a rule (``[game] title``), listed in ``titles.TITLES``.
"""

from attentive_snake.games.atari_versus.game import GAME

__all__ = ["GAME"]
