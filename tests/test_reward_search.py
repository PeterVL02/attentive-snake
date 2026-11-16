"""The reward search without the app: perturbation, selection, state files and the driver against a fake backend."""

import dataclasses
import json
from pathlib import Path

import pytest

from attentive_snake.games.registry import get_game
from attentive_snake.games.validation import spec_problems
from attentive_snake.reward_search.backend import FinalEvaluation, LaunchRefused
from attentive_snake.reward_search.member_config import member_tables
from attentive_snake.reward_search.naming import member_run, validate_search_name
from attentive_snake.reward_search.objective import objective_choices, objective_definition
from attentive_snake.reward_search.perturbation import MAX_DRAWS, perturbed_value, population_rewards
from attentive_snake.reward_search.reward_fields import searchable_fields, valid_reward
from attentive_snake.reward_search.searches import RewardSearches, SearchExists
from attentive_snake.reward_search.selection import winner
from attentive_snake.reward_search.settings import SearchSettings
from attentive_snake.reward_search.spec import SearchSpec
from attentive_snake.reward_search.state import state_from_json, state_to_json
from attentive_snake.reward_search.storage import SearchFiles

SNAKE = get_game("snake")
START = {"food": 1.0, "food_growth": 0.0, "death": -1.0, "win": 1.0, "starved": -1.0, "step": -0.005, "closer_to_food": 0.0, "farther_from_food": 0.0}


def always(_table) -> bool:
    return True


# --------------------------------------------------------------------------- perturbation


def test_member_zero_keeps_the_table_and_only_selected_fields_change_reproducibly():
    settings = SearchSettings(population=5, seed=3)
    tables = population_rewards(START, START, ("death", "step"), settings, 0, always)
    assert len(tables) == 5 and tables[0] == START
    for table in tables[1:]:
        assert {k for k in START if table[k] != START[k]} <= {"death", "step"}
        assert table != START
    assert tables == population_rewards(START, START, ("death", "step"), settings, 0, always)
    assert tables != population_rewards(START, START, ("death", "step"), settings, 1, always)  # another generation, other draws
    assert tables != population_rewards(START, START, ("death", "step"), dataclasses.replace(settings, seed=4), 0, always)


def test_moves_are_bounded_keep_their_sign_and_zero_uses_zero_scale():
    settings = SearchSettings(perturbation=5.0, max_change=0.5, zero_scale=0.02)
    rng = __import__("random").Random(0)
    for _ in range(500):
        death = perturbed_value(-1.0, -1.0, settings, rng)
        assert -1.5 <= death <= -0.5
        food = perturbed_value(1.0, 1.0, dataclasses.replace(settings, max_change=3.0), rng)
        assert 0.0 <= food <= 4.0  # may reach 0, never below
        zero = perturbed_value(0.0, 0.0, settings, rng)
        assert -0.01 <= zero <= 0.01
    crossing = SearchSettings(perturbation=3.0, max_change=3.0, allow_sign_change=True)
    assert any(perturbed_value(1.0, 1.0, crossing, rng) < 0 for _ in range(200))
    zeros = [perturbed_value(0.0, 0.0, SearchSettings(), rng) for _ in range(200)]
    assert min(zeros) < 0 < max(zeros)  # 0 has no sign to keep
    assert abs(perturbed_value(-1.0, -1.0, SearchSettings(perturbation=0.3), rng) + 1.0) <= 0.3  # one move: at most perturbation x size


def test_a_table_the_game_rejects_is_drawn_again_and_never_returned():
    lander = get_game("lunar_lander")
    start = {"scale": 0.01, "timeout": 0.0}
    settings = SearchSettings(population=8, perturbation=1.0, max_change=1.0)
    tables = population_rewards(start, start, ("scale",), settings, 0, lambda t: valid_reward(lander, t))
    assert all(table["scale"] > 0 for table in tables)
    with pytest.raises(ValueError, match=f"{MAX_DRAWS} draws"):
        population_rewards(START, START, ("death",), SearchSettings(), 0, lambda t: False)


# --------------------------------------------------------------------------- selection, objective, fields, names


def test_the_best_score_wins_ties_go_to_the_unchanged_member_and_failed_members_cannot_win():
    assert winner([1.0, 3.0, 2.0], [True] * 3, True) == 1
    assert winner([1.0, 3.0, 2.0], [True] * 3, False) == 0
    assert winner([3.0, 3.0, 2.0], [True] * 3, True) == 0
    assert winner([1.0, 3.0, 3.0], [True] * 3, True) == 1  # among changed members: the lowest index
    assert winner([1.0, 9.0], [True, False], True) == 0
    assert winner([None, 2.0], [True, True], True) == 1
    assert winner([None, None], [True, True], True) == 0  # nobody judged: the unchanged member carries on
    assert winner([None, 5.0], [False, False], True) is None


def test_the_objective_never_measures_the_reward_and_defaults_to_the_games_choice():
    keys = [definition.key for definition in objective_choices(SNAKE)]
    assert "mean_return" not in keys and "mean_length" not in keys and "games" not in keys
    assert {"mean_score", "win_rate"} <= set(keys)
    assert objective_definition(SNAKE, None).key == "mean_score"
    assert objective_definition(SNAKE, "mean_steps_to_win").higher_is_better is False
    with pytest.raises(ValueError, match="mean_return"):
        objective_definition(SNAKE, "mean_return")
    assert spec_problems(dataclasses.replace(SNAKE, search_objective="mean_return"), "snake")
    assert spec_problems(dataclasses.replace(SNAKE, search_objective="nope"), "snake")


def test_searchable_fields_are_the_float_coefficients():
    assert searchable_fields(SNAKE) == tuple(START)
    assert searchable_fields(get_game("atari")) == ("scale", "life_lost")  # not the clip switch


def test_names_and_settings_are_validated():
    assert member_run("s", 2, 3) == "s-g2-m3"
    for bad in ("", ".hidden", "a/b", "x" * 81):
        with pytest.raises(ValueError):
            validate_search_name(bad)
    for bad in ({"population": 1}, {"generations": 0}, {"steps_per_member": 0}, {"perturbation": 0.0}, {"zero_scale": -1.0}):
        with pytest.raises(ValueError):
            SearchSettings(**bad)
    with pytest.raises(ValueError):
        SearchSpec("s", "snake", "p", "c.pt", (), SearchSettings())


def test_member_tables_set_reward_length_and_a_final_evaluation_only():
    parent = {"game": {"name": "snake", "width": 6}, "reward": {"food": 1.0}, "train": {"total_env_steps": 9, "seed": 4}, "evaluation": {"seed": 7}}
    tables = member_tables(parent, {"food": 2.0}, SearchSettings(steps_per_member=500, evaluation_games=20))
    assert tables == {
        "game": {"name": "snake", "width": 6}, "reward": {"food": 2.0}, "train": {"total_env_steps": 500, "seed": 4},
        "evaluation": {"seed": 7, "interval": 500, "games": 20},
    }
    assert parent["train"]["total_env_steps"] == 9  # not changed in place


# --------------------------------------------------------------------------- the driver against a fake backend


class FakeBackend:
    """Runs as plain dicts: progress per run, final evaluations per run, and a log of what was queued."""

    def __init__(self) -> None:
        self.progress_of: dict[str, str] = {"base": "ended"}
        self.evaluations: dict[str, FinalEvaluation] = {}
        self.tables: dict[str, dict] = {"base": {"game": {"name": "snake"}, "reward": {"death": -1.0}, "train": {}}}
        self.forks: list[tuple] = []
        self.resumes: list[str] = []
        self.cancelled: list[str] = []
        self.refuse: str | None = None

    def run_tables(self, game, run):
        if run not in self.tables:
            raise LookupError(run)
        return self.tables[run]

    def enqueue_fork(self, game, parent, checkpoint, run, tables, description):
        if self.refuse:
            raise LaunchRefused(self.refuse)
        self.forks.append((parent, checkpoint, run, tables, description))
        self.progress_of[run] = "waiting"
        self.tables[run] = tables

    def enqueue_resume(self, game, run):
        self.resumes.append(run)
        self.progress_of[run] = "waiting"

    def progress(self, game, run):
        return self.progress_of.get(run, "missing")

    def last_training_evaluation(self, game, run):
        return self.evaluations.get(run)

    def cancel(self, game, run):
        self.cancelled.append(run)

    def finish(self, run: str, score: float | None, steps: int = 96) -> None:
        """The run ended with its final evaluation (at a whole update, a little short of its length of 100)."""
        self.progress_of[run] = "ended"
        self.evaluations[run] = FinalEvaluation(steps, f"{run}.pt", {"mean_score": score, "mean_return": 1e9})


def spec(name: str = "s", **settings) -> SearchSpec:
    values = {"population": 3, "generations": 2, "steps_per_member": 100} | settings
    return SearchSpec(name, "snake", "base", "base.pt", ("death", "step"), SearchSettings(**values))


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def searches(tmp_path: Path, backend: FakeBackend) -> RewardSearches:
    return RewardSearches(SearchFiles(tmp_path / "reward_searches"), backend)


def members(searches: RewardSearches, name: str = "s", generation: int = -1) -> list:
    return list(searches.get(name).generations[generation].members)


def test_a_search_runs_its_generations_and_follows_the_winner(searches: RewardSearches, backend: FakeBackend):
    state = searches.start(spec())
    assert state.start_reward == START | {"death": -1.0} and state.objective == "mean_score" and state.generations == ()
    searches.tick()  # plans generation 0, queues nothing yet
    assert [m.state for m in members(searches)] == ["planned"] * 3 and backend.forks == []
    searches.tick()  # queues its members
    assert [fork[2] for fork in backend.forks] == ["s-g0-m0", "s-g0-m1", "s-g0-m2"]
    assert {fork[:2] for fork in backend.forks} == {("base", "base.pt")}
    first = backend.forks[0][3]
    assert first["reward"] == state.start_reward and first["train"]["total_env_steps"] == 100 and first["evaluation"]["interval"] == 100
    assert "member 0 (unchanged reward)" in backend.forks[0][4]
    searches.tick()  # nothing new
    assert len(backend.forks) == 3 and [m.state for m in members(searches)] == ["waiting"] * 3

    backend.progress_of["s-g0-m0"] = "training"
    backend.finish("s-g0-m1", 5.0)
    backend.finish("s-g0-m2", 7.0)
    searches.tick()
    assert [m.state for m in members(searches)] == ["training", "done", "done"]
    backend.finish("s-g0-m0", 6.0)
    searches.tick()  # judges generation 0 and plans generation 1 from its winner
    state = searches.get("s")
    assert state.generations[0].winner == 2 and len(state.generations) == 2
    champion = state.generations[0].members[2]
    assert (champion.score, champion.checkpoint) == (7.0, "s-g0-m2.pt")
    assert state.generations[1].members[0].reward == champion.reward  # the winner's table, unchanged
    searches.tick()
    assert {fork[:2] for fork in backend.forks[3:]} == {("s-g0-m2", "s-g0-m2.pt")}
    for m, score in enumerate([4.0, 4.0, 3.0]):
        backend.finish(f"s-g1-m{m}", score)
    searches.tick()
    state = searches.get("s")
    assert state.status == "finished" and state.generations[1].winner == 0 and len(state.generations) == 2
    searches.tick()
    assert searches.get("s") == state and len(backend.forks) == 6  # a finished search does nothing more


def test_a_restart_queues_nothing_twice_and_an_early_end_is_resumed_once(tmp_path: Path, searches: RewardSearches, backend: FakeBackend):
    searches.start(spec())
    searches.tick()
    backend.progress_of["s-g0-m0"] = "waiting"  # queued just before the app stopped, state still says planned
    again = RewardSearches(SearchFiles(tmp_path / "reward_searches"), backend)  # a new app
    again.tick()
    assert [fork[2] for fork in backend.forks] == ["s-g0-m1", "s-g0-m2"]
    backend.progress_of["s-g0-m1"] = "ended"  # ended before its final evaluation (e.g. stopped with the app)
    again.tick()
    assert backend.resumes == ["s-g0-m1"] and members(again)[1].resumed and members(again)[1].state == "waiting"
    again.tick()
    backend.progress_of["s-g0-m1"] = "ended"
    again.tick()
    assert backend.resumes == ["s-g0-m1"] and members(again)[1].state == "failed" and "resume" in members(again)[1].error


def test_members_that_cannot_be_queued_or_vanish_fail_and_a_generation_without_survivors_ends_the_search(searches: RewardSearches, backend: FakeBackend):
    searches.start(spec())
    searches.tick()
    backend.refuse = "run 's-g0-m0' already exists"
    searches.tick()
    assert all(m.state == "failed" and "already exists" in m.error for m in members(searches))
    searches.tick()
    state = searches.get("s")
    assert state.status == "failed" and "generation 0" in state.error


def test_a_member_whose_queue_entry_vanished_fails(searches: RewardSearches, backend: FakeBackend):
    searches.start(spec(population=2))
    searches.tick()
    searches.tick()
    del backend.progress_of["s-g0-m1"]  # removed from the queue by hand
    backend.finish("s-g0-m0", 2.0)
    searches.tick()
    vanished = members(searches, generation=0)[1]
    assert vanished.state == "failed" and "Queue tab" in vanished.error
    assert searches.get("s").generations[0].winner == 0


def test_stop_cancels_unfinished_members_and_starting_is_validated(searches: RewardSearches, backend: FakeBackend):
    searches.start(spec())
    searches.tick()
    searches.tick()
    backend.finish("s-g0-m0", 1.0)
    searches.tick()
    stopped = searches.stop("s")
    assert stopped.status == "stopped" and sorted(backend.cancelled) == ["s-g0-m1", "s-g0-m2"]
    searches.tick()
    assert searches.get("s") == stopped and searches.stop("s") == stopped
    with pytest.raises(SearchExists, match="exists already"):
        searches.start(spec())
    backend.progress_of["t-g1-m2"] = "ended"
    with pytest.raises(SearchExists, match="t-g1-m2"):
        searches.start(spec("t"))
    with pytest.raises(ValueError, match="no reward coefficient"):
        searches.start(dataclasses.replace(spec("u"), fields=("death", "nope")))
    with pytest.raises(ValueError, match="objective"):
        searches.start(spec("u", objective="mean_return"))
    with pytest.raises(LookupError):
        searches.start(dataclasses.replace(spec("u"), parent="missing"))
    with pytest.raises(LookupError):
        searches.get("never")
    assert [state.spec.name for state in searches.all()] == ["s"]


def test_the_state_file_round_trips_and_is_replaced_atomically(tmp_path: Path, searches: RewardSearches, backend: FakeBackend):
    searches.start(spec(objective="win_rate"))
    searches.tick()
    state = searches.get("s")
    path = tmp_path / "reward_searches" / "s.json"
    data = json.loads(path.read_text())
    assert data["version"] == 1 and data["spec"]["settings"]["objective"] == "win_rate" and data["objective"] == "win_rate"
    assert state_from_json(state_to_json(state)) == state
    assert not list(path.parent.glob("*.tmp"))
    (path.parent / "broken.json").write_text("{")
    assert [s.spec.name for s in searches.all()] == ["s"]  # an unreadable file is left out
    searches.tick()  # and does not stop the others
    with pytest.raises(ValueError, match="cannot be read"):
        searches.get("broken")
