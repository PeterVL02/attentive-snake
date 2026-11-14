# Command-line cheat sheet

All commands run from the project root with `uv run ...`.

## Play the game yourself

```bash
uv run python -m attentive_snake.game.play --width 10 --height 10 --fps 8 --cell-size 30
```

Arrow keys or WASD steer, `R` restarts after a game over, `Esc` quits.

## Start a training run

```bash
# Defaults from configs/default.toml, run saved in runs/8x8
uv run python scripts/train.py --run-dir runs/8x8

# Your own config file
uv run python scripts/train.py --run-dir runs/mine --config configs/mine.toml

# Command-line overrides (new runs only; they win over the config file)
uv run python scripts/train.py --run-dir runs/small --width 6 --height 6 --num-envs 32 --seed 1 --steps 2000000

# Any single setting, a description, and keeping only the latest checkpoint
uv run python scripts/train.py --run-dir runs/lr-test --set train.learning_rate=1e-4 --set reward.step=0 \
    --description "lower LR, no living cost" --no-keep-checkpoints
```

- `--run-dir` defaults to `runs/default`. A directory that already holds checkpoints is refused; use `--resume`.
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
uv run python scripts/train.py --resume runs/8x8

# Train longer: raise the budget (the saved config.toml is updated too)
uv run python scripts/train.py --resume runs/8x8 --steps 10000000

# With a note, and from now on keep only the latest checkpoint (saved to config.toml)
uv run python scripts/train.py --resume runs/8x8 --description "more steps" --no-keep-checkpoints
```

`--resume` uses the run's own `config.toml`; `--config`, `--set`, `--width`, `--height`, `--num-envs`, `--seed`
and `--fork` are rejected with it. To continue with other settings, start a new run from a checkpoint (next section).

## Continue a run under a new name with changed settings (fork)

```bash
# New run runs/8x8-lowlr starting from the latest checkpoint of runs/8x8, with a lower learning rate
uv run python scripts/train.py --run-dir runs/8x8-lowlr --fork runs/8x8 --set train.learning_rate=1e-4

# From a specific checkpoint, with other rewards, a new budget and a note
uv run python scripts/train.py --run-dir runs/8x8-food2 --fork runs/8x8 --fork-checkpoint checkpoint_0000500000.pt \
    --set reward.food=2 --steps 2000000 --description "food worth more"

# From an edited copy of the parent's config
cp runs/8x8/config.toml configs/8x8-variant.toml   # edit it, then:
uv run python scripts/train.py --run-dir runs/8x8-variant --fork runs/8x8 --config configs/8x8-variant.toml
```

- The new run loads the parent's weights and optimizer state, then counts its own env steps from 0 up to its own
  `total_env_steps` (by default the parent's), with its own learning-rate schedule starting from `learning_rate`.
  Adam's moment estimates are carried over even when the learning rate changes (they do not depend on it).
- Its settings are the parent's `config.toml` (or `--config`) plus `--set`, `--steps`, `--num-envs`, `--seed`,
  `--keep-checkpoints`. Everything under `[reward]` and `[train]`, `env.num_envs` and
  `game.max_steps_without_food` may change. All of `[model]`, `game.width`, `game.height`, `game.initial_length`,
  `env.history_length` and `env.history_actions` must equal the parent's (the weights would not fit, or the input
  would change meaning); a change is rejected naming the field, before anything is written.
- `--run-dir` is required and must be a new run (a directory with checkpoints is refused). `--fork-checkpoint`
  takes a checkpoint file name of the parent (or its path); the default is the parent's latest.
- The parent is never modified. The new run's `run.json` records the parent run, checkpoint and its env steps.

## Watch a trained agent

```bash
# Latest checkpoint of the run, greedy actions
uv run python scripts/watch.py runs/8x8

# With the attention heatmap, a specific layer, sampled actions, faster
uv run python scripts/watch.py runs/8x8 --attention --layer 0 --sample --fps 20

# A specific checkpoint
uv run python scripts/watch.py runs/8x8 --checkpoint runs/8x8/checkpoint_0000100000.pt
```

Other flags: `--cell-size N`, `--frame N`. `Esc` or closing the window quits.
`--layer` picks the transformer block (negative counts from the last); `--frame` picks the history frame (0 = oldest, -1 = newest).

| model kind | `--attention` shows |
| --- | --- |
| `transformer`, `hybrid` | heatmap of what the class token attends to on the board (`--layer` applies, `--frame` must stay -1/0) |
| `temporal_transformer` | heatmap for one of the history frames (`--layer`, `--frame`) |
| `frame_attention` | one weight per history frame, printed in the window title (`--layer`) |
| `cnn` | nothing: there is no attention, so `--attention` is refused |

## Where configs live

| What | Where |
| --- | --- |
| Default experiment (every setting except the rewards, commented) | `configs/default.toml` |
| Your own experiments | copy it to e.g. `configs/mine.toml`, pass with `--config` |
| Named reward schemes; `default` holds the default reward values | `configs/rewards/<name>.toml` (flat file of the `[reward]` keys, no header) |
| The exact config of one run | `runs/<name>/config.toml` |

An experiment file has the sections `[game] [reward] [model] [train] [env]`. A missing key takes its default; an unknown key is an error.
A config without a `[reward]` section gets the default reward values (those of `configs/rewards/default.toml`).
To use another reward scheme in an experiment, copy its lines under a `[reward]` header in the experiment file
(only the keys that differ are needed), or pass single values with `--set reward.food=2`.

## What a run directory contains

```
runs/<name>/
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

| `model.kind` | idea | suggested `[env]` |
| --- | --- | --- |
| `transformer` | each board cell is a token | `history_length = 1` |
| `cnn` | convolutional baseline, no attention | `history_length = 1` |
| `hybrid` | conv stem, then transformer over cells | `history_length = 1` |
| `temporal_transformer` | attention over the cells of several frames (plus action tokens) | `history_length = 4`, `history_actions = true` |
| `frame_attention` | CNN per frame, attention across frames | `history_length = 4`, `history_actions = true` |

The non-temporal kinds see stacked frames only as extra input channels. Which `[model]` fields each kind uses
(for example `cnn` ignores `num_heads`) is listed by the app's config page.

## Draw a model diagram from Python

```bash
uv run python -c "
from attentive_snake.config import load_experiment
from attentive_snake.factory import make_env, make_policy
from attentive_snake.viz import diagram, save_diagram

config = load_experiment('configs/default.toml')
policy = make_policy(config, make_env(config))
print(diagram(policy))                       # text tree
save_diagram(policy, 'model.html')           # .html, .mmd, .md or .txt
"
```

## Run the tests

```bash
uv run pytest -q
```

## Start the web app

```bash
uv run python -m attentive_snake.app
```

Serves http://127.0.0.1:8000: configure, start and resume training, continue a run under a new name with changed
settings, watch an agent with the attention heatmap, view the architecture and edit reward schemes.
