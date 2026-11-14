# Adding a game

A game is one directory under `src/attentive_snake/games/` plus its configs. Nothing
else is edited: the registry finds the directory, and training, resuming, forking,
the config schema, reward schemes, watching and the app's backend all work through
one contract, the `GameSpec`. The test `tests/test_games.py` proves this with a
dummy game that lives only in `tests/dummy_games/walk/` (worth reading as a minimal
example). Snake in `games/snake/` is a full grid game written from scratch; Lunar Lander
in `games/lunar_lander/` is a Gymnasium game wrapped in a few small modules.

## 1. Files

```
src/attentive_snake/games/<name>/
    __init__.py        exposes GAME (and nothing heavy: no pygame, no torch)
    game.py            GAME = GameSpec(...)                         (convention)
    config.py          the rules dataclass  -> the [game] section   (convention)
    reward.py          the reward dataclass -> the [reward] section (convention)
    env.py             the GameEnv                                  (convention)
    ...                anything else the game needs (engine, encoder, view, reward table)

configs/<name>/default.toml           the default experiment of the game
configs/<name>/rewards/default.toml   the default reward values (the "default" scheme)
configs/<name>/rewards/<scheme>.toml  more reward schemes (optional)
```

`<name>` uses only `a-z`, `0-9` and `_`. It is the package name, `GAME.name`, the value
of `[game] name` in configs, and the directory name under `configs/` and `runs/`.

Only `__init__.py` with `GAME` is required by the registry; the other module names
are a convention. Keep each module to one job, and keep game code free of torch
(the environment side speaks numpy only).

## 2. The `GameSpec` (`games/spec.py`)

```python
GAME = GameSpec(
    name="lunar_lander",
    title="Lunar Lander",
    description="One or two plain sentences for the app.",
    observation_kind="vector",        # "grid" | "vector" | "image"
    view_kind="image",                # "board" | "image"
    rules_type=LanderRules,           # [game] section (without "name")
    reward_type=LanderReward,         # [reward] section
    make_env=make_env,                # (rules, reward, seed) -> GameEnv
    locked_rules={},                  # {rules field: reason} a fork must keep
    reward_table=None,                # optional (reward, example_score) -> rows
    observation_labels=("x", "y", ...),  # optional, vector games: a name per number of a frame
    episode_metrics=None,             # extension point for issue #4; leave None
)
```

| Field | Meaning |
| --- | --- |
| `observation_kind` | `"grid"`: a small board, shape `(F, H, W)`, every cell a token (Snake). `"vector"`: a flat list of numbers, shape `(F,)`. `"image"`: screen pixels, shape `(F, H, W)`. Policies declare which kinds they read (`BasePolicy.observation_kinds`); every policy reads `"grid"` and `"vector"` (each picks its input stage by kind, `models/input_stages/`); none reads `"image"` yet. |
| `view_kind` | Which watch payload `GameEnv.view()` returns (section 5). |
| `rules_type`, `reward_type` | Frozen, flat dataclasses (section 4). |
| `make_env` | Builds one `GameEnv` (section 3). Called once per parallel env with a derived seed. |
| `locked_rules` | Rules that change the observation's shape or meaning (Snake: `width`, `height`, `initial_length`). A continued run (fork) must keep them; `game.name`, all of `[model]`, `env.history_length` and `env.history_actions` are always locked. |
| `observation_labels` | Vector games only, optional: one short name per number of one frame, in order (Lunar Lander: `x`, `y`, `vx`, ...). Shown next to the attention over the numbers (`WatchSession.info["element_names"]`); without it they are called `number 0`, `number 1`, .... The registry builds one default env to check the count. |
| `reward_table` | What a scheme pays in typical situations, for the app's Rewards tab: rows `{"situation": str, "reward": float, "breakdown": {term: float}}`. `None`: no preview (`/api/rewards/preview` answers 404). |
| `episode_metrics` | Reserved for per-game evaluation statistics (steps to win, steps between apples, ...): `(finished EpisodeSummary list) -> {metric: float}`. Nothing calls it yet. |

The registry checks the spec when it scans (`games/validation.py`) and raises
`GameRegistryError` naming the package if anything is missing or wrong, including an
import error in the package.

## 3. The environment: `GameEnv` (`env/types.py`)

```python
class GameEnv(Protocol):
    observation_shape: tuple[int, ...]   # ONE frame: (F, H, W) for grid/image, (F,) for vector
    num_actions: int                     # discrete actions 0 .. num_actions-1

    def reset(self, seed: int | None = None) -> np.ndarray: ...   # float32, observation_shape
    def step(self, action: int) -> EnvStep: ...
    def view(self) -> dict[str, Any]: ...                         # payload of view_kind
```

`EnvStep(observation, reward, terminated, truncated, event, score)`:

- `observation`: float32 array of `observation_shape`.
- `terminated`: the game's rules ended the episode (death, win). `truncated`: a limit
  cut it short (starvation, time limit). The trainer treats both as the end.
- `event`: what happened in this step in the game's words (`"moved"`, `"crashed"`, ...).
  On the last step of an episode it is the episode's outcome and ends up in
  `EpisodeSummary.outcome`.
- `score`: the game's headline number for the episode so far (Snake: food eaten;
  Lunar Lander: e.g. the return). It becomes `EpisodeSummary.score` and the
  `mean_score` metric.

The env does not auto-reset and knows nothing about history. `view()` is only ever
called after a `reset()` (Gymnasium games cannot render before one); everything that
watches a game resets it first. You get for free:

- `HistoryEnv` (`env/history_env.py`): the last `env.history_length` frames stacked
  along axis 0, oldest first, and with `env.history_actions = true` `num_actions`
  one-hot action channels appended to every frame (constant over `H, W` for grids and
  images, `num_actions` extra numbers for vectors).
- `SyncVecEnv` (`env/vec_env.py`): `env.num_envs` copies, auto-reset, per-env seeds
  (`env_seed(seed, i)`), episode return/length/score/outcome tracking.
- The PPO trainer, checkpoints, resume, fork, metrics, `run.json`, the CLI and the
  app's backend.

A Gymnasium game is a thin adapter: `reset` calls `env.reset(seed=seed)`, `step` maps
`(obs, reward, terminated, truncated, info)` to `EnvStep`, and `view()` renders. Make the
Gymnasium env without a render mode and switch it to `"rgb_array"` in the first `view()`
(see `games/lunar_lander/env.py`): training envs never call `view()`, so they never pay
for rendering. Import Gymnasium inside the function that makes the env, so finding the
game stays cheap.

## 4. Config dataclasses and files

Both dataclasses are frozen and flat; field types are `int`, `float`, `bool`, `str`,
`Path` or any of them `| None`. Every field needs `metadata={"help": "..."}`: it is
the description in the app and in the schema. The rules dataclass may not have a field
called `name` (reserved for `[game] name`). Validate values in `__post_init__`.

```python
@dataclass(frozen=True)
class LanderRules:
    gravity: float = field(default=-10.0, metadata={"help": "Downward acceleration."})
```

`configs/<name>/default.toml` must name the game and list every field of `[game]`,
`[model]`, `[train]` and `[env]` (fields that default to `None` are commented out,
TOML has no null), but no `[reward]` section:

```toml
[game]
name = "lunar_lander"
gravity = -10.0

[model]
...
```

`configs/<name>/rewards/default.toml` lists every reward field with exactly the
dataclass defaults (flat, no header). `tests/test_config.py` checks both files for
every registered game, so a new game fails the suite until they exist.

## 5. Watching: the view payloads (`views/`)

`GameEnv.view()` returns the payload of the game's `view_kind`. Both have `width` and
`height`; other keys must not be one of the generic watch-frame keys (`score steps
event done heatmap frame_weights element_weights caption episode view_kind`), because the watch frame
merges the payload into its top level.

`"board"` (`views/board.py`, build with `board_payload(width, height, body, target)`):

```json
{"width": 8, "height": 8, "snake": [[3, 4], [2, 4]], "food": [6, 1]}
```

Cells are `[x, y]`, x to the right, y down. `snake` is the player body, head first;
`food` the single target cell or `null`. The app and `scripts/watch.py` draw this.

`"image"` (`views/image.py`, build with `image_payload(rgb)` from an `(H, W, 3)` uint8
array, e.g. Gymnasium's `render_mode="rgb_array"`):

```json
{"width": 600, "height": 400, "png": "<base64 PNG, no data: prefix>"}
```

The app's watch tab shows it (`static/js/image_view.js`), and `scripts/watch.py` shows it
in a pygame window (`views/image_window.py`).

A whole watch frame (`WatchFrame.to_dict()`, what `/api/watch` returns per step) is the
payload plus `view_kind`, `score`, `steps`, `event`, `done`, `heatmap` (the
observation's `(H, W)` attention for grid observations, else `null`),
`frame_weights` (one per history frame, for `frame_attention`), `element_weights`
(vector observations: one weight in [0, 1] per number of the selected frame, in the
order of `observation_labels`, else `null`), `caption`, `episode`. A kind's
`attention_view` of `"grid"` or `"frames"` means element weights on a vector game.

## 6. Check it

```bash
uv run pytest -q
uv run python scripts/train.py --game <name> --run-dir runs/<name>/try --steps 20000
uv run python scripts/watch.py runs/<name>/try
```
