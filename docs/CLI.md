# Command-line cheat sheet

All commands run from the project root with `uv run ...`.

Every run belongs to one game: `snake` (the default) or `lunar_lander` (Gymnasium's LunarLander-v3). Each game
has its own default config (`configs/<game>/default.toml`), its own reward schemes (`configs/<game>/rewards/`) and
its own runs directory (`runs/<game>/`). `--game NAME` picks the game for a new run. Resuming, forking and watching read the game from the run's
`config.toml`.

## Play Snake yourself

```bash
uv run python -m attentive_snake.games.snake.play --width 10 --height 10 --fps 8 --cell-size 30
```

Arrow keys or WASD steer, `R` restarts after a game over, `Esc` quits.

## Start a training run

```bash
# Defaults from configs/snake/default.toml, run saved in runs/snake/8x8
uv run python scripts/train.py --run-dir runs/snake/8x8

# The game's defaults and runs/<game>/default (the game is snake unless --game or the --config file says otherwise)
uv run python scripts/train.py --game snake

# Your own config file
uv run python scripts/train.py --run-dir runs/snake/mine --config configs/mine.toml

# Command-line overrides (new runs only; they win over the config file)
uv run python scripts/train.py --run-dir runs/snake/small --width 6 --height 6 --num-envs 32 --seed 1 --steps 2000000

# Any single setting, a description, and keeping only the latest checkpoint
uv run python scripts/train.py --run-dir runs/snake/lr-test --set train.learning_rate=1e-4 --set reward.step=0 \
    --description "lower LR, no living cost" --no-keep-checkpoints
```

```bash
# Lunar Lander with its defaults (mlp, 32 envs, 2M steps: a few minutes on a CPU), run saved in runs/lunar_lander/mlp
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/mlp

# Another model kind; the temporal kinds want history
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/tt --set model.kind=temporal_transformer \
    --set env.history_length=4 --set env.history_actions=true
```

- `--game NAME` (new runs only) picks `configs/<NAME>/default.toml` and the default run directory `runs/<NAME>/default`.
  Without it the game is the `--config` file's `[game] name`, else `snake`; a `--game` that differs from the config's
  game is refused.
- `--run-dir` defaults to `runs/<game>/default`. Keep runs in `runs/<game>/` so the app finds them. A directory
  that already holds checkpoints is refused; use `--resume`.
- `--width` / `--height` are shortcuts for `--set game.width=...` / `--set game.height=...` (games with a board).
- `--steps` sets `train.total_env_steps` (the training budget over all parallel environments).
- `--set section.key=value` changes one setting (repeatable; the value is read as TOML, e.g. `1e-4`, `true`, `"cpu"`).
  It cannot set a value to None; use a config file for that.
- `--description TEXT` is saved in the run's `run.json` (see below) and shown in the app.
- `--no-keep-checkpoints` deletes the older checkpoints of the run each time a new one is saved, so only the
  latest remains (saved as `train.keep_checkpoints = false`). Deleted checkpoints can no longer be watched or
  continued from. `--keep-checkpoints` (the default) keeps them all.

## Resume or extend a run

```bash
# Continue from the latest checkpoint up to the run's total_env_steps
uv run python scripts/train.py --resume runs/snake/8x8

# Train longer: raise the budget (the saved config.toml is updated too)
uv run python scripts/train.py --resume runs/snake/8x8 --steps 10000000

# With a note, and from now on keep only the latest checkpoint (saved to config.toml)
uv run python scripts/train.py --resume runs/snake/8x8 --description "more steps" --no-keep-checkpoints
```

`--resume` uses the run's own `config.toml` (including its game); `--game`, `--config`, `--set`, `--width`,
`--height`, `--num-envs`, `--seed` and `--fork` are rejected with it. To continue with other settings, start a new run from a checkpoint (next section).

## Continue a run under a new name with changed settings (fork)

```bash
# New run runs/snake/8x8-lowlr starting from the latest checkpoint of runs/snake/8x8, with a lower learning rate
uv run python scripts/train.py --run-dir runs/snake/8x8-lowlr --fork runs/snake/8x8 --set train.learning_rate=1e-4

# From a specific checkpoint, with other rewards, a new budget and a note
uv run python scripts/train.py --run-dir runs/snake/8x8-food2 --fork runs/snake/8x8 --fork-checkpoint checkpoint_0000500000.pt \
    --set reward.food=2 --steps 2000000 --description "food worth more"

# From an edited copy of the parent's config
cp runs/snake/8x8/config.toml configs/8x8-variant.toml   # edit it, then:
uv run python scripts/train.py --run-dir runs/snake/8x8-variant --fork runs/snake/8x8 --config configs/8x8-variant.toml
```

- The new run loads the parent's weights and optimizer state, then counts its own env steps from 0 up to its own
  `total_env_steps` (by default the parent's), with its own learning-rate schedule starting from `learning_rate`.
  Adam's moment estimates are carried over even when the learning rate changes (they do not depend on it).
- Its settings are the parent's `config.toml` (or `--config`) plus `--set`, `--steps`, `--num-envs`, `--seed`,
  `--keep-checkpoints`. Everything under `[reward]` and `[train]`, `env.num_envs` and the game rules the game does
  not lock (Snake: `game.max_steps_without_food`) may change. The game (`game.name`), all of `[model]`,
  `env.history_length`, `env.history_actions` and the rules the game locks (Snake: `game.width`, `game.height`,
  `game.initial_length`) must equal the parent's (the weights would not fit, or the input would change meaning);
  a change is rejected naming the field, before anything is written.
- `--run-dir` is required and must be a new run (a directory with checkpoints is refused). `--fork-checkpoint`
  takes a checkpoint file name of the parent (or its path); the default is the parent's latest.
- The parent is never modified. The new run's `run.json` records the parent run, checkpoint and its env steps.

## Watch a trained agent

```bash
# Latest checkpoint of the run, greedy actions
uv run python scripts/watch.py runs/snake/8x8

# With the attention heatmap, a specific layer, sampled actions, faster
uv run python scripts/watch.py runs/snake/8x8 --attention --layer 0 --sample --fps 20

# A specific checkpoint
uv run python scripts/watch.py runs/snake/8x8 --checkpoint runs/snake/8x8/checkpoint_0000100000.pt
```

```bash
# Lunar Lander: the game's own picture; --attention puts the weight of each input number in the window title
uv run python scripts/watch.py runs/lunar_lander/tt --attention --scale 0.75
```

The window shows the run's game (it reads the game from the run's `config.toml`); games whose view is a board
(Snake) are drawn cell by cell (`--cell-size N`), games whose view is an image (Lunar Lander) show the picture
(`--scale F`). Other flag: `--frame N`. `Esc` or closing the window quits.
`--layer` picks the transformer block (negative counts from the last); `--frame` picks the history frame (0 = oldest, -1 = newest).

| model kind | `--attention` shows (board games) | (vector games such as Lunar Lander) |
| --- | --- | --- |
| `transformer`, `hybrid` | heatmap of what the class token attends to on the board (`--layer` applies, `--frame` must stay -1/0) | one weight per input number in the window title, e.g. `x 0.12, y 1.00, ...` |
| `temporal_transformer` | heatmap for one of the history frames (`--layer`, `--frame`) | one weight per number of one history frame (`--frame`) |
| `frame_attention` | one weight per history frame, printed in the window title (`--layer`) | the same |
| `cnn`, `mlp` | nothing: there is no attention, so `--attention` is refused | the same |

## Where configs live

| What | Where |
| --- | --- |
| Default experiment of a game (every setting except the rewards, commented) | `configs/<game>/default.toml` |
| Your own experiments | copy it to e.g. `configs/mine.toml`, pass with `--config` |
| Named reward schemes of a game; `default` holds its default reward values | `configs/<game>/rewards/<name>.toml` (flat file of the `[reward]` keys, no header) |
| The exact config of one run | `runs/<game>/<name>/config.toml` |

An experiment file has the sections `[game] [reward] [model] [train] [env]`. `[game] name` says which game it is
(a file without it is a Snake config, so configs written before there were several games still load); the other
`[game]` keys are that game's rules and `[reward]` its reward coefficients. A missing key takes its default; an
unknown key is an error. A config without a `[reward]` section gets the game's default reward values (those of
`configs/<game>/rewards/default.toml`).
To use another reward scheme in an experiment, copy its lines under a `[reward]` header in the experiment file
(only the keys that differ are needed), or pass single values with `--set reward.food=2`.

## What a run directory contains

```
runs/<game>/<name>/
    config.toml                 the full resolved config (written at start)
    model.html                  interactive architecture diagram
    metrics.jsonl               one JSON line per logged update (env_steps, scores, losses, ...)
    run.json                    history: one entry per start / resume / fork (see below)
    checkpoint_<steps>.pt       policy + optimizer + env-step counter
```

Metrics are logged every `train.log_interval` updates and at the last update; checkpoints every `train.checkpoint_interval` updates and at the end.
Lower both for short experiments.

`run.json` is not configuration; it is a log of what was done to the run:

```json
{"events": [
  {"action": "fork", "env_steps_at_action": 0, "description": "lower LR",
   "parent": {"run": "8x8", "checkpoint": "checkpoint_0000500000.pt", "env_steps": 500000},
   "timestamp": "2026-11-14T10:00:00+00:00"},
  {"action": "resume", "env_steps_at_action": 250000, "description": null, "timestamp": "..."}
]}
```

`action` is `start`, `resume` or `fork`; `env_steps_at_action` is the run's own step count at that moment;
`parent` exists only on a fork event. Runs created before `run.json` existed have no history.

## Choosing a model kind and history

Set `kind` in `[model]` and `history_length` / `history_actions` in `[env]`.

| `model.kind` | idea on a board (Snake) | on numbers (Lunar Lander) | suggested `[env]` |
| --- | --- | --- | --- |
| `transformer` | each board cell is a token | each number is a token | `history_length = 1` |
| `cnn` | convolutional baseline, no attention | an MLP (nothing to convolve); recommended | `history_length = 1` |
| `hybrid` | conv stem, then transformer over cells | per-number MLP stem, then transformer over numbers | `history_length = 1` |
| `temporal_transformer` | attention over the cells of several frames (plus action tokens) | attention over every number of several frames (plus action tokens) | `history_length = 4`, `history_actions = true` |
| `frame_attention` | CNN per frame, attention across frames | MLP per frame, attention across frames | `history_length = 4`, `history_actions = true` |
| `mlp` | flattened board, weak baseline | plain MLP; recommended | `history_length = 1` |

The non-temporal kinds see stacked frames only as extra input channels (extra values per number). Which `[model]` fields each kind uses
(for example `cnn` ignores `num_heads`) is listed by the app's config page.

## Draw a model diagram from Python

```bash
uv run python -c "
from attentive_snake.config import load_experiment
from attentive_snake.factory import make_env, make_policy
from attentive_snake.viz import diagram, save_diagram

config = load_experiment('configs/snake/default.toml')
policy = make_policy(config, make_env(config))
print(diagram(policy))                       # text tree
save_diagram(policy, 'model.html')           # .html, .mmd, .md or .txt
"
```

## Move runs from the old layout

Runs made before there were several games sit directly in `runs/<name>/`. They are all Snake runs; this moves them
to `runs/snake/<name>/` (only directories that contain a `config.toml`; nothing is ever overwritten, and nothing is
moved if any target already exists). Running it again does nothing.

```bash
uv run python scripts/migrate_runs.py --dry-run     # show what would be moved
uv run python scripts/migrate_runs.py               # move them (--runs-dir DIR for another runs root)
```

## Run the tests

```bash
uv run pytest -q
```

## Start the web app

```bash
uv run python -m attentive_snake.app [--port 8000] [--runs-dir runs] [--configs-dir configs]
```

Serves http://127.0.0.1:8000: configure, start and resume training, continue a run under a new name with changed
settings, watch an agent with its attention, view the architecture and edit reward schemes.
