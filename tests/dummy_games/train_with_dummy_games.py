"""``scripts/train.py`` with the dummy games of this directory registered, for training subprocesses started by tests.

The app launches training as a subprocess of its ``train_script``; tests that train a dummy game
through the app point ``AppPaths.train_script`` here, so the subprocess finds the game as well.
"""

import runpy
from pathlib import Path

import attentive_snake.games as games_package

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE.parents[1] / "scripts" / "train.py"

if __name__ == "__main__":
    games_package.__path__.append(str(HERE))
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
