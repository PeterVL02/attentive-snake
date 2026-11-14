"""Names of the numbers of one frame of a vector observation, for showing attention over them."""

from attentive_snake.games.spec import GameSpec
from attentive_snake.observation_spec import ObservationSpec


def element_names(game: GameSpec, spec: ObservationSpec) -> list[str] | None:
    """The game's ``observation_labels`` (else ``"number 0"``, ...) for vector observations; ``None`` otherwise.

    Only the game's own numbers are named (``spec.base_channels``), not the action numbers.
    """
    if spec.kind != "vector":
        return None
    if game.observation_labels is not None:
        return list(game.observation_labels)
    return [f"number {i}" for i in range(spec.base_channels)]
