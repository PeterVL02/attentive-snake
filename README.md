# Attentive Snake

Train neural networks to play games with reinforcement learning (PPO or DQN), and look
inside them: which board cells or input numbers a transformer attends to, how the
architecture is put together, and how different models and reward schemes compare.

The project contains two games (Snake and Gymnasium's Lunar Lander), a
game-agnostic headless parallel RL environment layer, six selectable policy
architectures, two training algorithms (PPO and DQN), an architecture-diagram tool, and a local web app
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
went from a mean training score of about -170 to about +150 in one test run. 200
counts as solved. The greedy evaluation at 2M steps of that run (seed 0, 50 games)
landed every game with a mean score of 256; the same run with
`train.bootstrap_truncated = false` (how the time limit was treated before)
hovered until the time limit in 49 of 50 games (mean score 1). One seed each, not a
benchmark (see Caveats).

## The app

| Tab | What it is for |
|---|---|
| **New training** | Pick a model, fill in its settings (each with a description), choose the board, reward scheme and training settings, add an optional description, and start a run. The state representation (history length, past actions) is pre-set from the chosen model. |
| **Runs** | Live status, score chart, log, config and history (descriptions, where a run was continued from) of a run. Its **evaluations** (see below): the latest one's statistics, a chart of any evaluation metric, and "Run evaluation now" for any checkpoint. Stop it, resume it with an optional new step total and description, or **continue with changes**: a new run under a new name that starts from a chosen checkpoint with edited rewards and training settings. Open any checkpoint in the watch tab. |
| **Architecture** | Diagram, text tree and parameter count for the config in the form or for an existing run. |
| **Watch & attention** | Watch a checkpoint play. Board games show the attention heatmap over the board; picture games such as Lunar Lander show the frame with attention bars per input number. |
| **Rewards** | Create and edit named reward schemes, with a live table of what a scheme pays in each situation. |
| **CLI cheat sheet** | The contents of [docs/CLI.md](docs/CLI.md). |
| **Compare** | Plot the score curves or an evaluation metric of several runs on one chart, with their descriptions and latest/best evaluation numbers, and list the config fields that differ. |

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

## Training algorithms

`[train] algorithm` picks how a run learns; its own settings are the section of
the same name. Every model kind works with both: the network's trunk is the
same, only its head differs, so attention and architecture diagrams work alike.

| Algorithm | What it is here | Head |
|---|---|---|
| `ppo` (default) | Proximal policy optimisation: on-policy actor-critic. Plays rollouts with its own sampled actions, computes GAE advantages and takes a few epochs of clipped-surrogate updates on them, then throws them away. | policy (action probabilities) + value |
| `dqn` | Double DQN: off-policy and value-based. Plays epsilon-greedily, stores every transition in a replay buffer, and takes gradient steps on random batches of it with a Huber loss against a target network (the online network picks the next action, the target network values it). | one Q-value per action |

```bash
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/dqn --set train.algorithm=dqn
```

or pick DQN in the app's New-training form, which then shows DQN's settings and
fills in the game's DQN defaults (`configs/<game>/algorithms/dqn.toml`, laid over
`configs/<game>/default.toml`). Evaluation and watching are greedy for both (the
most likely action, or the highest Q-value); the Watch tab shows the action
probabilities and state value of a PPO run, or the Q-values of a DQN run.

DQN settings that matter most: `learning_rate` (in `[train]`), `train_frequency`
(env steps per gradient step, whatever `num_envs` is: the replay ratio),
`target_update_interval`, `epsilon_fraction` / `epsilon_end` (exploration) and
`buffer_size`. One DQN iteration is one step of every parallel game, so
`num_envs` only changes how experience is gathered, not how much is learned
per step.

In one test (one seed each, CPU, 500k env steps on Lunar Lander with the default
`mlp`): DQN with its defaults reached a greedy evaluation mean score of 203 (41
of 50 games landed) in under 3 minutes; PPO with its defaults and the same
budget ended at -29 (it needs its full 2M steps). See the Caveats for what this
does and does not say.

## Configs and runs

Everything game-specific is per game: `[game] name` in a config says which game
it is (a config without it is a Snake config).

- `configs/<game>/default.toml` lists every setting with a comment, except the
  reward values. Copy it to make your own experiment config. It trains with PPO;
  `configs/<game>/algorithms/dqn.toml` holds the game's DQN defaults (the `[dqn]`
  section and the shared values that differ), used when a new run asks for DQN.
- `configs/<game>/rewards/*.toml` are the game's named reward schemes.
  `configs/<game>/rewards/default.toml` is the one place the game's default
  reward values are written down: a config without a `[reward]` section gets
  exactly those. Choosing a scheme for a run copies its values into the run;
  editing the scheme later does not change existing runs.
- Each run lives in `runs/<game>/<name>/` with its resolved `config.toml`, checkpoints,
  `metrics.jsonl`, `evaluations.jsonl` (if it was evaluated), `run.json` (history of starts, resumes and forks, with
  descriptions), `train.log` (when started from the app) and `model.html`.
  `runs/` is not tracked by git. Runs from before there were several games sit
  in `runs/<name>/`; `uv run python scripts/migrate_runs.py` moves them to
  `runs/snake/<name>/` (add `--dry-run` to see the plan first).
- `train.keep_checkpoints = false` (`--no-keep-checkpoints`, or the checkbox in
  the app) keeps only a run's latest checkpoint to save disk space; older
  checkpoints are deleted and can no longer be watched or continued from.

## Evaluation

The training curve comes from the exploring policy (PPO samples its actions, DQN
takes random ones with probability epsilon). An **evaluation** shows what the model
can do: it plays complete games always taking its best action (the most likely one,
or the highest Q-value), on games of its own (a fixed seed, separate from
training), and reports statistics per game:

| Game | Its own statistics (plus scores, episode lengths, returns and how games ended) |
|---|---|
| Snake | win rate; mean and median steps to win (won games only); mean and median steps between apples (from the start or the previous apple to each apple; the steps after a game's last apple do not count) |
| Lunar Lander | landing, crash, fly-off and timeout rates; mean steps to land (landed games only) |

A statistic that cannot be computed (steps to win when no game was won) is N/A.

- **While training:** `[evaluation] interval` (env steps, 0 = off; on by default in
  `configs/<game>/default.toml`, off for runs whose config predates it), `games` and
  `seed`. Each evaluation saves a checkpoint at that step and appends one line to
  `runs/<game>/<name>/evaluations.jsonl`. Evaluating does not change training: a run
  with evaluation ends with exactly the same weights as one without.
- **Afterwards:** `uv run python scripts/evaluate.py runs/snake/8x8 [--checkpoint FILE]
  [--games X] [--seed S] [--save]`, or "Run evaluation now" in the Runs tab. The same
  checkpoint, number of games and seed give exactly the same numbers, so a recorded
  evaluation can be reproduced; both say whether the result matches one.

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
- **A continued run (fork) cannot change the game, algorithm, network, board or observation.**
  The game, `train.algorithm`, all of `[model]`, the board size, `initial_length`, `history_length` and
  `history_actions` must match the parent, because its weights would not fit or
  would see a different input. It keeps the parent's Adam moments and starts its
  own learning-rate schedule from `learning_rate`, counting its steps from 0.
- **Extending a run raises the learning rate.** The linear decay is re-stretched
  over the new total, so the rate jumps up at the resume point. A resumed run is
  not bit-identical to an uninterrupted one.
- **Starvation (Snake) is treated as a terminal state; the time limit (Lunar
  Lander) is not, by default.** `train.bootstrap_truncated = true` (Lunar Lander's
  default) credits the step that hit a limit with the value of the state it was cut
  at; `false` (Snake's default, and every run made before the setting existed)
  counts nothing after the cut, so the values of states near the limit are
  underestimated. Snake's starvation limit stays a real ending: it has its own
  reward and the steps since the last apple are not in the observation.
- **Lunar Lander's training scores stay below "solved".** Training episodes play
  sampled actions; with the default mlp they average about 900 steps and many end
  by the 1000-step time limit rather than coming to rest, so the training curve
  flattens near 150 even when the greedy evaluation lands every game. Without
  `train.bootstrap_truncated` the greedy policy itself hovered until the time limit.
  "landed" means the lander came to rest anywhere, not necessarily between the flags.
- **Attention kinds on Lunar Lander are for looking, not for scores.** With 8
  input numbers, `mlp` and `cnn` learn faster; the transformer kinds are there to
  show which inputs the decision draws on.
- **Keep dropout at 0.** PPO's probability ratio becomes noisy with dropout on.
- **DQN's replay buffer is not saved.** Checkpoints hold the online and target
  networks, the optimizer and the step count. A resumed or forked DQN run starts
  with an empty buffer and takes no gradient step until it holds
  `dqn.learning_starts` transitions again; a fork also restarts exploration from
  `epsilon_start` (its step count starts at 0).
- **DQN's replay buffer lives in memory**: about 2 x `buffer_size` x (numbers in
  one observation) x 4 bytes, since each transition keeps its observation and the
  next one as float32 (Snake 8x8 with 100k transitions: ~150 MB; history frames
  multiply it).
- **A DQN run cannot be forked into a PPO run (or back).** The weights end in the
  other algorithm's head and the optimizer state belongs to it.
- **One-seed comparisons of algorithms are anecdotes.** DQN and PPO react very
  differently to their settings and budgets; the numbers above are single runs
  with each algorithm's defaults, not tuned against each other.
- **Architecture diagrams show containment and order, not true data flow.**
  Skip connections and branching are not visible. The rendered diagram in the
  app loads Mermaid from a CDN and falls back to the text tree offline.
- **The time axis in Compare is an estimate** built from recorded steps per
  second; pauses between resumes are not counted.
- **Runs created before `metrics.jsonl` existed have no score chart.**
- **Evaluations reproduce exactly only on the same device.** GPU and CPU (and
  different GPUs) can differ in the last bits of a result, which now and then
  changes a greedy action. Evaluations record their device; a check on another
  device says so when the numbers differ.
- **`keep_checkpoints = false` makes older evaluations irreproducible.** Each
  evaluation saves a checkpoint, but the next save deletes it. The Runs tab marks
  such evaluations "(deleted)"; keep checkpoints if you want to re-check them.
- **An evaluation is a sample.** 100 games give a win rate to within a few
  percentage points; compare checkpoints with the same seed and number of games.
- **Evaluation games have a step cap** (`evaluation.max_episode_steps`, default
  10,000): a game still going then (a Snake without a starvation limit circling
  forever) ends with the outcome `step_limit`.

## Development

```bash
uv run pytest -q
```

The code is organised so that each module has one job; see [CLAUDE.md](CLAUDE.md)
for a map of the packages and the contracts between them.
