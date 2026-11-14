"""Which input stage reads which observation kind, per role a policy needs.

Roles (each builder is ``(StageInput, ModelConfig) -> Stage``):

* ``TOKEN_STAGES``: one token per position (grid cell / vector element), ``d_model`` wide.
* ``FEATURE_STAGES``: the whole input -> one ``d_model`` vector.
* ``FRAME_STAGES``: one frame -> one ``d_model`` vector (same as ``FEATURE_STAGES`` but for
  the names grid checkpoints already use).
* ``HYBRID_STAGES``: a local/per-position stem, then one token per position.

Adding an observation kind (e.g. ``"image"``: patches as tokens, convolutions as stem)
means one new module next to ``grid.py`` / ``vector.py`` and an entry in each table it serves.
"""

from collections.abc import Callable, Mapping

from attentive_snake.models.config import ModelConfig
from attentive_snake.models.input_stages import grid, vector
from attentive_snake.models.input_stages.stage import Stage, StageInput

type StageBuilder = Callable[[StageInput, ModelConfig], Stage]

TOKEN_STAGES: Mapping[str, StageBuilder] = {"grid": grid.token_stage, "vector": vector.token_stage}
FEATURE_STAGES: Mapping[str, StageBuilder] = {"grid": grid.feature_stage, "vector": vector.feature_stage}
FRAME_STAGES: Mapping[str, StageBuilder] = {"grid": grid.frame_stage, "vector": vector.feature_stage}
HYBRID_STAGES: Mapping[str, StageBuilder] = {"grid": grid.hybrid_stage, "vector": vector.hybrid_stage}


def readable_kinds(*tables: Mapping[str, StageBuilder]) -> tuple[str, ...]:
    """Observation kinds every one of ``tables`` has a stage for (a policy's ``observation_kinds``)."""
    first, *rest = tables
    return tuple(kind for kind in first if all(kind in table for table in rest))
