"""Finding games: every sub-package of ``attentive_snake.games`` that exposes ``GAME: GameSpec``.

There is no central list: dropping a directory ``games/<name>/`` with an ``__init__.py``
that defines ``GAME`` is enough. The scan runs once per process (``reload_games`` repeats
it, e.g. in tests that add a games directory to ``attentive_snake.games.__path__``).
A sub-package that fails to import or exports an incomplete ``GAME`` raises
``GameRegistryError`` naming the package.
"""

import functools
import importlib
import pkgutil

import attentive_snake.games as games_package
from attentive_snake.games.spec import GameSpec
from attentive_snake.games.validation import spec_problems

DEFAULT_GAME = "snake"


class GameRegistryError(RuntimeError):
    """A game package is broken (import error, missing or incomplete ``GAME``)."""


def game_names() -> list[str]:
    """Names of all games, alphabetical."""
    return sorted(_games())


def get_game(name: str) -> GameSpec:
    """The game called ``name``; ``ValueError`` listing the known games if there is none."""
    games = _games()
    if name not in games:
        raise ValueError(f"unknown game {name!r}; known games: {', '.join(sorted(games))}.")
    return games[name]


def reload_games() -> None:
    """Forget the scan, so the next lookup scans the games directories again."""
    _games.cache_clear()


@functools.cache
def _games() -> dict[str, GameSpec]:
    found = {}
    for module in pkgutil.iter_modules(games_package.__path__):
        if module.ispkg:
            found[module.name] = _load(module.name)
    return found


def _load(package_name: str) -> GameSpec:
    qualified = f"{games_package.__name__}.{package_name}"
    try:
        module = importlib.import_module(qualified)
    except Exception as error:
        raise GameRegistryError(f"game package {qualified!r} cannot be imported: {error!r}") from error
    spec = getattr(module, "GAME", None)
    if not isinstance(spec, GameSpec):
        raise GameRegistryError(f"game package {qualified!r} must define GAME = GameSpec(...) in its __init__.py.")
    problems = spec_problems(spec, package_name)
    if problems:
        raise GameRegistryError(f"game package {qualified!r}: {'; '.join(problems)}.")
    return spec
