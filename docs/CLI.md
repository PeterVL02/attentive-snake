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
```

- `--run-dir` defaults to `runs/default`. A directory that already holds checkpoints is refused; use `--resume`.
- `--steps` sets `train.total_env_steps` (the training budget over all parallel environments).
- Every other setting is edited in the TOML file (see "Where configs live").

## Resume or extend a run

```bash
# Continue from the latest checkpoint up to the run's total_env_steps
uv run python scripts/train.py --resume runs/8x8

# Train longer: raise the budget (the saved config.toml is updated too)
uv run python scripts/train.py --resume runs/8x8 --steps 10000000
```

`--resume` uses the run's own `config.toml`; `--config`, `--width`, `--height`, `--num-envs` and `--seed` are rejected with it.
To change other settings of a run, edit `runs/<name>/config.toml` before resuming.

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
| Default experiment (every setting, commented) | `configs/default.toml` |
| Your own experiments | copy it to e.g. `configs/mine.toml`, pass with `--config` |
| Named reward schemes | `configs/rewards/<name>.toml` (flat file of the `[reward]` keys, no header) |
| The exact config of one run | `runs/<name>/config.toml` |

An experiment file has the sections `[game] [reward] [model] [train] [env]`. A missing key takes its default; an unknown key is an error.
To use a reward scheme in an experiment, copy its lines under `[reward]` in the experiment file.

## What a run directory contains

```
runs/<name>/
    config.toml                 the full resolved config (written at start)
    model.html                  interactive architecture diagram
    metrics.jsonl               one JSON line per logged update (env_steps, scores, losses, ...)
    checkpoint_<steps>.pt       policy + optimizer + env-step counter
```

Metrics are logged every `train.log_interval` updates and at the last update; checkpoints every `train.checkpoint_interval` updates and at the end.
Lower both for short experiments.

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

Serves http://127.0.0.1:8000: configure, start and resume training, watch an agent with the attention heatmap,
view the architecture and edit reward schemes.
