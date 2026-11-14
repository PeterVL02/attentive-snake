# Attentive Snake

Train neural networks to play Snake with reinforcement learning (PPO), and look
inside them: which board cells a transformer attends to, how the architecture is
put together, and how different models and reward schemes compare.

The project contains a classic Snake game, a headless parallel RL environment,
five selectable policy architectures, a PPO trainer, an architecture-diagram
tool, and a local web app that ties it all together.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14. A CUDA GPU is used
automatically when available.

```bash
uv sync
uv run python -m attentive_snake.app
```

Open <http://127.0.0.1:8000>. The app only listens on your own machine.

## The app

| Tab | What it is for |
|---|---|
| **New training** | Pick a model, fill in its settings (each with a description), choose the board, reward scheme and training settings, add an optional description, and start a run. The state representation (history length, past actions) is pre-set from the chosen model. |
| **Runs** | Live status, score chart, log, config and history (descriptions, where a run was continued from) of a run. Stop it, resume it with an optional new step total and description, or **continue with changes**: a new run under a new name that starts from a chosen checkpoint with edited rewards and training settings. Open any checkpoint in the watch tab. |
| **Architecture** | Diagram, text tree and parameter count for the config in the form or for an existing run. |
| **Watch & attention** | Watch a checkpoint play, with the attention heatmap drawn over the board. |
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

The two temporal kinds need history: set `history_length` (for example 4) and
`history_actions = true` under `[env]`. The app does this for you.

## Configs and runs

- `configs/default.toml` lists every setting with a comment, except the reward
  values. Copy it to make your own experiment config.
- `configs/rewards/*.toml` are named reward schemes. `configs/rewards/default.toml`
  is the one place the default reward values are written down: a config without
  a `[reward]` section gets exactly those. Choosing a scheme for a run copies its
  values into the run; editing the scheme later does not change existing runs.
- Each run lives in `runs/<name>/` with its resolved `config.toml`, checkpoints,
  `metrics.jsonl`, `run.json` (history of starts, resumes and forks, with
  descriptions), `train.log` (when started from the app) and `model.html`.
  `runs/` is not tracked by git.
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
- **A continued run (fork) cannot change the network, board or observation.**
  All of `[model]`, the board size, `initial_length`, `history_length` and
  `history_actions` must match the parent, because its weights would not fit or
  would see a different input. It keeps the parent's Adam moments and starts its
  own learning-rate schedule from `learning_rate`, counting its steps from 0.
- **Extending a run raises the learning rate.** The linear decay is re-stretched
  over the new total, so the rate jumps up at the resume point. A resumed run is
  not bit-identical to an uninterrupted one.
- **Starvation is treated as a terminal state** when estimating values, which
  introduces a small bias.
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
