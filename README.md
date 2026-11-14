# Attentive Snake

Train neural networks to play games with reinforcement learning (PPO), and look
inside them: which board cells or input numbers a transformer attends to, how the
architecture is put together, and how different models and reward schemes compare.

The project contains two games (Snake and Gymnasium's Lunar Lander), a
game-agnostic headless parallel RL environment layer, six selectable policy
architectures, a PPO trainer, an architecture-diagram tool, and a local web app
that ties it all together.

## Games

| Game | Observation | What the model sees |
|---|---|---|
| `snake` (default) | grid | The whole board: one cell per position, channels head / body / food. |
| `lunar_lander` | vector | 8 numbers: x, y, vx, vy, angle, angular velocity, left and right leg contact (Gymnasium's `LunarLander-v3`, discrete actions). Watched as the game's own picture. |

A game is one directory under `src/attentive_snake/games/` plus its configs;
nothing else is edited. See [docs/ADDING_A_GAME.md](docs/ADDING_A_GAME.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14. A CUDA GPU is used
automatically when available.

```bash
uv sync
uv run python -m attentive_snake.app
```

Open <http://127.0.0.1:8000>. The app only listens on your own machine.

### Lunar Lander quick start

```bash
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/mlp   # about 2-3 minutes on a CPU
uv run python scripts/watch.py runs/lunar_lander/mlp
```

The default (`configs/lunar_lander/default.toml`: `mlp`, 32 envs, 2M steps)
went from a mean score of about -170 to about +145 in one test run. 200 counts
as solved; the default does not get there (see Caveats).

## The app

| Tab | What it is for |
|---|---|
| **New training** | Pick a model, fill in its settings (each with a description), choose the board, reward scheme and training settings, add an optional description, and start a run. The state representation (history length, past actions) is pre-set from the chosen model. |
| **Runs** | Live status, score chart, log, config and history (descriptions, where a run was continued from) of a run. Stop it, resume it with an optional new step total and description, or **continue with changes**: a new run under a new name that starts from a chosen checkpoint with edited rewards and training settings. Open any checkpoint in the watch tab. |
| **Architecture** | Diagram, text tree and parameter count for the config in the form or for an existing run. |
| **Watch & attention** | Watch a checkpoint play. Board games show the attention heatmap over the board; picture games such as Lunar Lander show the frame with attention bars per input number. |
| **Rewards** | Create and edit named reward schemes, with a live table of what a scheme pays in each situation. |
| **CLI cheat sheet** | The contents of [docs/CLI.md](docs/CLI.md). |
| **Compare** | Plot the score curves of several runs on one chart, with their descriptions, and list the config fields that differ. |

Everything the app does is also available from the terminal: see the
[CLI cheat sheet](docs/CLI.md) for playing the game yourself, training,
resuming and watching.

## Models

Chosen with `kind` in the `[model]` section of a config (or the dropdown in the app).

| Kind | What it does | Attention view |
|---|---|---|
| `transformer` | One token per board cell; attention over the board. | Board heatmap |
| `cnn` | Residual convolutions with a linear readout. Baseline. | None |
| `hybrid` | A conv stem embeds each cell with its neighbourhood, then transformer blocks. | Board heatmap |
| `temporal_transformer` | Attention over the cells of the last K frames plus one token per past action. | Board heatmap per frame |
| `frame_attention` | A CNN turns each of the last K frames into one vector; attention runs across frames. | Weight per frame |
| `mlp` | A plain multilayer perceptron on the flattened observation. Baseline for vector games. | None |

Every kind reads both grids and vectors; only the input stage changes. On a
vector observation (Lunar Lander) every number becomes a token where a board cell
would, the convolutions become fully connected layers (a convolution over an
arbitrary order of numbers means nothing), and the attention view is one weight per
input number instead of a board heatmap. `mlp`
is the recommended kind for vector games; the app marks the recommended kinds
for the chosen game.

The two temporal kinds need history: set `history_length` (for example 4) and
`history_actions = true` under `[env]`. The app does this for you.

## Configs and runs

Everything game-specific is per game: `[game] name` in a config says which game
it is (a config without it is a Snake config).

- `configs/<game>/default.toml` lists every setting with a comment, except the
  reward values. Copy it to make your own experiment config.
- `configs/<game>/rewards/*.toml` are the game's named reward schemes.
  `configs/<game>/rewards/default.toml` is the one place the game's default
  reward values are written down: a config without a `[reward]` section gets
  exactly those. Choosing a scheme for a run copies its values into the run;
  editing the scheme later does not change existing runs.
- Each run lives in `runs/<game>/<name>/` with its resolved `config.toml`, checkpoints,
  `metrics.jsonl`, `run.json` (history of starts, resumes and forks, with
  descriptions), `train.log` (when started from the app) and `model.html`.
  `runs/` is not tracked by git. Runs from before there were several games sit
  in `runs/<name>/`; `uv run python scripts/migrate_runs.py` moves them to
  `runs/snake/<name>/` (add `--dry-run` to see the plan first).
- `train.keep_checkpoints = false` (`--no-keep-checkpoints`, or the checkbox in
  the app) keeps only a run's latest checkpoint to save disk space; older
  checkpoints are deleted and can no longer be watched or continued from.

## Caveats

- **Stopping the app stops the trainings it started.** For a long run that
  should outlive the app, start it from the terminal.
- **Stopping a run loses progress since its last checkpoint.**
- **Runs started from a terminal show as idle in the app.** Do not resume from
  the app a run that is currently training in a terminal.
- **Pure transformers are slow to start learning.** In short tests the kinds
  with a convolutional stage (`hybrid`, `cnn`, `frame_attention`) began scoring
  much sooner. These were single short runs, not a benchmark.
- **History adds no information in principle.** Snake is fully observable from
  one frame, so the temporal models are for studying attention over time, not
  for better scores, and they train slower.
- **A model is tied to its board size.** A checkpoint cannot be used on a
  different grid.
- **A continued run (fork) cannot change the game, network, board or observation.**
  The game, all of `[model]`, the board size, `initial_length`, `history_length` and
  `history_actions` must match the parent, because its weights would not fit or
  would see a different input. It keeps the parent's Adam moments and starts its
  own learning-rate schedule from `learning_rate`, counting its steps from 0.
- **Extending a run raises the learning rate.** The linear decay is re-stretched
  over the new total, so the rate jumps up at the resume point. A resumed run is
  not bit-identical to an uninterrupted one.
- **Starvation (Snake) and the time limit (Lunar Lander) are treated as terminal
  states** when estimating values, which introduces a small bias: the value of a
  state just before the cut does not include what would have come after.
- **Lunar Lander's default stalls below "solved".** The default mlp learns to
  hover and land slowly, and many episodes end by the 1000-step time limit (mean
  length about 900). A negative `reward.timeout` or a longer run are the
  first things to try. "landed" means the lander came to rest anywhere, not
  necessarily between the flags.
- **Attention kinds on Lunar Lander are for looking, not for scores.** With 8
  input numbers, `mlp` and `cnn` learn faster; the transformer kinds are there to
  show which inputs the decision draws on.
- **Keep dropout at 0.** PPO's probability ratio becomes noisy with dropout on.
- **Architecture diagrams show containment and order, not true data flow.**
  Skip connections and branching are not visible. The rendered diagram in the
  app loads Mermaid from a CDN and falls back to the text tree offline.
- **The time axis in Compare is an estimate** built from recorded steps per
  second; pauses between resumes are not counted.
- **Runs created before `metrics.jsonl` existed have no score chart.**

## Development

```bash
uv run pytest -q
```

The code is organised so that each module has one job; see [CLAUDE.md](CLAUDE.md)
for a map of the packages and the contracts between them.
