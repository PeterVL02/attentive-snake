"""Names: a search's own (also its file name) and its member runs', ``<search>-g<generation>-m<member>``."""

from attentive_snake.run_dir import validate_run_name

MAX_NAME_LENGTH = 80


def validate_search_name(name: str) -> str:
    """``name`` unchanged if it can name a search and its member runs, else ``ValueError``."""
    validate_run_name(name)
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"a search name may have at most {MAX_NAME_LENGTH} characters.")
    return name


def member_run(search: str, generation: int, member: int) -> str:
    """The run name of member ``member`` of generation ``generation`` (both counted from 0)."""
    return f"{search}-g{generation}-m{member}"


def all_member_runs(search: str, generations: int, population: int) -> list[str]:
    """Every run name a search of this size may create, in order."""
    return [member_run(search, g, m) for g in range(generations) for m in range(population)]
