"""All reward searches: start one, read them, stop one, and ``tick`` (advance every running search one step and save it).

Thread-safe: the app's routes and its ticker thread share one ``RewardSearches``; every method
holds one lock. In process, without HTTP::

    searches = RewardSearches(SearchFiles(searches_dir(runs_dir)), backend)
    searches.start(SearchSpec("s", "snake", "base", "checkpoint_0000100000.pt", ("step", "death"), SearchSettings()))
    searches.tick()            # call regularly (the app: every second); the state is in searches.get("s")
"""

import dataclasses
import logging
import threading

from attentive_snake.games.registry import get_game
from attentive_snake.reward_search.backend import TrainingBackend
from attentive_snake.reward_search.driver import advance
from attentive_snake.reward_search.naming import all_member_runs, validate_search_name
from attentive_snake.reward_search.objective import objective_definition
from attentive_snake.reward_search.reward_fields import full_reward, require_searchable, valid_reward
from attentive_snake.reward_search.spec import SearchSpec
from attentive_snake.reward_search.state import SearchState
from attentive_snake.reward_search.storage import SearchFiles

_log = logging.getLogger(__name__)


class SearchExists(RuntimeError):
    """A search, or a run one of its members would create, already has this name."""


class RewardSearches:
    def __init__(self, files: SearchFiles, backend: TrainingBackend) -> None:
        self._files = files
        self._backend = backend
        self._lock = threading.Lock()

    def start(self, spec: SearchSpec) -> SearchState:
        """Save a new running search; its first generation is planned on the next ``tick``.

        ``ValueError`` for an invalid name, game, field or objective; ``LookupError`` if the parent
        run does not exist; ``SearchExists`` if the name or a member run's name is taken.
        """
        validate_search_name(spec.name)
        game = get_game(spec.game)
        require_searchable(game, spec.fields)
        objective = objective_definition(game, spec.settings.objective)
        with self._lock:
            if self._files.exists(spec.name):
                raise SearchExists(f"a reward search named {spec.name!r} exists already; choose another name.")
            taken = [run for run in all_member_runs(spec.name, spec.settings.generations, spec.settings.population) if self._backend.progress(spec.game, run) != "missing"]
            if taken:
                raise SearchExists(f"the search would create runs that exist or wait in the queue already: {', '.join(taken)}.")
            start_reward = full_reward(game, self._backend.run_tables(spec.game, spec.parent).get("reward", {}))
            state = SearchState(spec, objective.key, bool(objective.higher_is_better), start_reward)
            self._files.save(state)
            return state

    def get(self, name: str) -> SearchState:
        """``LookupError`` if there is none, ``ValueError`` if its name is invalid or its file unreadable."""
        with self._lock:
            return self._files.load(name)

    def all(self) -> list[SearchState]:
        """Every search whose file can be read, newest first."""
        with self._lock:
            states = [state for state in map(self._readable, self._files.names()) if state is not None]
        return sorted(states, key=lambda state: state.created_at, reverse=True)

    def stop(self, name: str) -> SearchState:
        """Stop a running search: its waiting member runs leave the queue, training ones are stopped. ``LookupError`` if there is none."""
        with self._lock:
            state = self._files.load(name)
            if state.status != "running":
                return state
            for generation in state.generations:
                for member in generation.members:
                    if not member.ended:
                        self._backend.cancel(state.spec.game, member.run)
            state = dataclasses.replace(state, status="stopped")
            self._files.save(state)
            return state

    def tick(self) -> None:
        """Advance every running search one step and save the ones that changed; one broken search does not stop the others."""
        with self._lock:
            for name in self._files.names():
                try:
                    self._advance(name)
                except Exception:
                    _log.exception("reward search %s: step failed", name)

    def _advance(self, name: str) -> None:
        state = self._files.load(name)
        if state.status != "running":
            return
        game = get_game(state.spec.game)
        advanced = advance(state, self._backend, lambda table: valid_reward(game, table))
        if advanced != state:
            self._files.save(advanced)

    def _readable(self, name: str) -> SearchState | None:
        try:
            return self._files.load(name)
        except (LookupError, ValueError):
            return None
