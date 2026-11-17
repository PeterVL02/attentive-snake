"""What the API says about reward searches: the form's rows for a game, a search in short, and a search in full."""

from typing import Any

from attentive_snake.config.schema import field_rows
from attentive_snake.games.spec import GameSpec
from attentive_snake.reward_search.objective import objective_choices
from attentive_snake.reward_search.reward_fields import searchable_fields
from attentive_snake.reward_search.settings import SearchSettings
from attentive_snake.reward_search.state import SearchState, state_to_json


def search_form(game: GameSpec) -> dict[str, Any]:
    """``{"game", "settings": [rows], "fields": [rows]}``: the settings (``objective`` offering the game's choices, its own
    as default) and the game's searchable reward coefficients, as schema rows (``config.schema.field_rows``)."""
    settings = field_rows(SearchSettings)
    for row in settings:
        if row["name"] == "objective":
            row.update(choices=[definition.key for definition in objective_choices(game)], default=game.search_objective, nullable=False)
    searchable = set(searchable_fields(game))
    return {"game": game.name, "settings": settings, "fields": [row for row in field_rows(game.reward_type) if row["name"] in searchable]}


def search_summary(state: SearchState) -> dict[str, Any]:
    """Name, game, parent, status, objective, how many generations are judged of how many, and the latest winner's score."""
    judged = [generation for generation in state.generations if generation.winner is not None]
    best = judged[-1].members[judged[-1].winner] if judged else None
    return {
        "name": state.spec.name,
        "game": state.spec.game,
        "parent": state.spec.parent,
        "status": state.status,
        "error": state.error,
        "created_at": state.created_at,
        "objective": state.objective,
        "fields": list(state.spec.fields),
        "generations_judged": len(judged),
        "generations": state.spec.settings.generations,
        "best_score": None if best is None else best.score,
        "best_run": None if best is None else best.run,
    }


def search_detail(state: SearchState) -> dict[str, Any]:
    """The summary and the whole state (spec, start reward, every generation and member)."""
    return {**search_summary(state), "state": state_to_json(state)}
