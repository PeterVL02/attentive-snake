"""Where reward searches are kept: one JSON file per search in ``<runs root>/reward_searches/<name>.json``.

The directory holds no run directories (no ``config.toml``), so the runs and games listings never
mistake it for a game, as with the training queue's ``queue.json``. Files are replaced atomically
(written beside, then renamed): a crash never leaves half a file.
"""

import json
import os
from pathlib import Path

from attentive_snake.reward_search.naming import validate_search_name
from attentive_snake.reward_search.state import SearchState, state_from_json, state_to_json

DIRECTORY_NAME = "reward_searches"
SUFFIX = ".json"


def searches_dir(runs_dir: Path) -> Path:
    return Path(runs_dir) / DIRECTORY_NAME


class SearchFiles:
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    def path(self, name: str) -> Path:
        """The file of search ``name`` (``ValueError`` for a name that is not a valid search name)."""
        return self._directory / f"{validate_search_name(name)}{SUFFIX}"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def names(self) -> list[str]:
        """Names of the saved searches, alphabetical."""
        if not self._directory.is_dir():
            return []
        return sorted(path.name.removesuffix(SUFFIX) for path in self._directory.glob(f"*{SUFFIX}") if path.is_file())

    def load(self, name: str) -> SearchState:
        """``LookupError`` if there is no such search, ``ValueError`` if its file cannot be read."""
        path = self.path(name)
        if not path.is_file():
            raise LookupError(f"there is no reward search named {name!r}.")
        try:
            return state_from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"the file of reward search {name!r} cannot be read: {error}") from error

    def save(self, state: SearchState) -> None:
        path = self.path(state.spec.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps(state_to_json(state), indent=1), encoding="utf-8")
        os.replace(temporary, path)
