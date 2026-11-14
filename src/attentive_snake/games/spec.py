"""``GameSpec``: the single contract a game package satisfies to be found and used everywhere.

A game is a sub-package ``attentive_snake.games.<name>`` exposing a module-level
``GAME: GameSpec``; ``games.registry`` finds it by scanning, nothing else needs editing.
Its configs live in ``configs/<name>/default.toml`` and ``configs/<name>/rewards/*.toml``,
its runs in ``runs/<name>/``. See docs/ADDING_A_GAME.md for the full recipe.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from attentive_snake.env.types import GameEnv
from attentive_snake.episode_metrics import GameMetrics
from attentive_snake.observation_spec import ObservationKind
from attentive_snake.views import ViewKind

type MakeEnv = Callable[[Any, Any, int | None], GameEnv]
type RewardTable = Callable[[Any, int], list[dict[str, Any]]]


@dataclass(frozen=True)
class GameSpec:
    """Everything the generic code needs to know about one game.

    Attributes:
        name: Identifier and directory name (``[a-z0-9_]``), e.g. ``"snake"``; equals the
            sub-package name. Written as ``[game] name`` in experiment configs.
        title: Human-readable name, e.g. ``"Snake"``.
        description: One or two plain sentences for the app.
        observation_kind: ``"grid"``, ``"vector"`` or ``"image"`` (see ``observation_spec``).
        view_kind: ``"board"`` or ``"image"``: the payload ``GameEnv.view()`` returns
            (see ``attentive_snake.views``).
        rules_type: Frozen flat dataclass of the game's rules: the ``[game]`` section
            (without ``name``, which is reserved). Every field needs ``metadata["help"]``
            and a TOML-friendly type (int, float, bool, str, Path, or ``| None``).
        reward_type: Frozen flat dataclass of the reward coefficients: the ``[reward]``
            section; same field rules.
        make_env: ``(rules, reward, seed) -> GameEnv``: one environment producing one
            frame per observation. History, action channels, vectorisation, auto-reset
            and episode statistics are added generically around it.
        locked_rules: ``{rules field: reason}`` for the rules a continued run (fork) must
            keep, because they change the observation's shape or meaning.
        reward_table: Optional ``(reward config, example_score) -> rows``: what a reward
            scheme pays in typical situations, rows ``{"situation", "reward",
            "breakdown": {term: value}}``. ``None``: the app shows no preview.
        observation_labels: Optional names of the numbers of one frame of a vector game, in
            order (Lunar Lander: ``"x"``, ``"y"``, ...), one per element of
            ``GameEnv.observation_shape``; shown next to the attention over them. ``None``:
            no names (displays number them). Only for ``observation_kind == "vector"``.
        episode_metrics: Optional ``GameMetrics`` (``attentive_snake.episode_metrics``): the
            game's own evaluation statistics (Snake: win rate, steps to win, steps between
            apples), computed from ``EpisodeRecord``s on top of the generic ones every game
            gets (scores, lengths, outcome counts). ``None``: only the generic statistics.
    """

    name: str
    title: str
    description: str
    observation_kind: ObservationKind
    view_kind: ViewKind
    rules_type: type
    reward_type: type
    make_env: MakeEnv
    locked_rules: Mapping[str, str] = field(default_factory=dict)
    reward_table: RewardTable | None = None
    observation_labels: tuple[str, ...] | None = None
    episode_metrics: GameMetrics | None = None
