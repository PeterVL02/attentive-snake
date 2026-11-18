# Attentive Snake

Train neural networks to play games with reinforcement learning (PPO or DQN), and look
inside them: which board cells or input numbers a transformer attends to, how the
architecture is put together, and how different models and reward schemes compare.

The project contains five games (Snake, Lunar Lander, Atari, MinAtar and two-player Atari by self-play), a game-agnostic
headless parallel environment layer, twelve model kinds, two training algorithms, an
architecture-diagram tool and a local web app that ties it all together. Everything the app
does can also be done from the terminal.

More detail:

- [docs/CLI.md](docs/CLI.md): every command, flag and setting (also shown in the app).
- [docs/ADDING_A_GAME.md](docs/ADDING_A_GAME.md): how to add a game (one directory plus its configs).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14. A CUDA GPU is used automatically
when available.

```bash
uv sync
uv run python -m attentive_snake.app
```

Open <http://127.0.0.1:8000>. The app only listens on your own machine.

From the terminal:

```bash
uv run python scripts/train.py --run-dir runs/snake/8x8                              # Snake, the defaults
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/mlp   # about 2-3 minutes on a CPU
uv run python scripts/watch.py runs/lunar_lander/mlp
uv run python scripts/evaluate.py runs/lunar_lander/mlp
```

Resuming, forking (continuing a run under a new name with changed settings), playing Snake
yourself and the rest are in [docs/CLI.md](docs/CLI.md).

## Games

| Game | Observation | What the model sees |
|---|---|---|
| `snake` (default) | grid | The whole board: one cell per position, channels head / body / food. |
| `lunar_lander` | vector | 8 numbers: position, velocity, angle, angular velocity, leg contacts (Gymnasium's `LunarLander-v3`). Watched as the game's own picture. |
| `atari` | image | The console's screen, shrunk to 84 x 84 grayscale, the last 4 frames. One game for six titles. |
| `minatar` | grid | MinAtar's 10 x 10 grid, one 0/1 channel per kind of object. One game for five titles. |
| `atari_versus` | image | Two-player Atari: the shared screen as for `atari`, plus one plane per frame saying which player the model is. Trained by self-play. |

Each game has its settings in `configs/<game>/default.toml`, with a comment on every field.

### Snake

Besides the board size, `initial_length` and the starvation limit there are two optional apple
rules, both off by default: `game.food_count = N` keeps up to N apples on the board, and
`game.food_distribution = "center"` places new apples near the middle (`food_spread` in cells).
Neither is locked on fork, so a run trained with easier apple rules can be continued on the
standard game (a curriculum). Scores under different apple rules are not comparable.

**Measured (8x8, PPO, 819,200 env steps, greedy evaluation of 100 games; one seed, not a benchmark).**
The default `transformer` learned nothing in that budget with any apple rule (mean score 0.05-0.14).
`jepa_attention` with 3 apples reached 36.5. Forking that run onto the standard game gave 18.7 / 26.7 /
31.9 at 82k / 164k / 246k steps, while `jepa_attention` trained from scratch on the standard game was
still at 0.05. So the apple-rule curriculum got it going.

### Lunar Lander

Four discrete actions by default, or two throttles with `game.continuous = true` (see
[Continuous actions](#continuous-actions)). `game.hide_velocities = true` replaces the three velocity
numbers by zeros, so a model has to infer them from several frames.

**Measured (the defaults: `mlp`, 32 envs, 2M env steps; one seed, not a benchmark).** The greedy evaluation
at 2M steps (50 games) landed every game with a mean score of 256 (200 counts as solved). The same run with
`train.bootstrap_truncated = false` hovered until the time limit in 49 of 50 games (mean score 1). The
training curve flattens near 150 even when the greedy policy lands every game: training plays sampled
actions and many of those games end at the 1000-step time limit. "Landed" means the lander came to rest,
not necessarily between the flags.

### Atari

```bash
uv run python scripts/train.py --game atari --run-dir runs/atari/pong                       # Pong, PPO, cnn
uv run python scripts/train.py --game atari --run-dir runs/atari/venture --set game.title=venture
```

`game.title` picks one of `pong` (default, the easy one), `ms_pacman` (dense rewards), and four
sparse-reward games: `montezuma_revenge`, `pitfall`, `private_eye`, `venture`. In short:

- Every title uses the same adapter and, with `game.full_action_space = true` (the default), the same 18
  actions, so a network has the same shape for every title.
- A step is 4 screen frames (`frame_skip`). The console repeats the previous action with probability
  `sticky_actions` (0.25).
- The training reward is the game's real points (`reward.clip = true` reduces them to their sign), and PPO runs
  scale them with `ppo.normalize_returns`, on by default for Atari only; `mean_score` everywhere is the real score.
  Losing a life does not end the game.
- Every model kind except `resnet` and `mlp` reads pictures; the default `cnn` with `d_model = 512` is the DQN
  paper's network.
- DQN's replay buffer is small here (16000 transitions take about 3.6 GB). DQN on Atari has not been measured.

**Measured (Pong, PPO, `cnn`, the defaults of that time, so `reward.clip = true`, with `--steps 1800000`; one seed, not a benchmark).** On an RTX 5080
with 8 envs: about 1,250 env steps/s, 24 minutes. The greedy policy lost every point (-21) until 1M steps,
then took off: -19.8 at 1.4M and -17.4 at 1.8M, with long rallies. Pong is usually won after 5-10M steps;
that run was not made.

**Measured (Ms. Pac-Man, PPO, `cnn`, 1.8M env steps; one seed, not a benchmark).** Greedy evaluation of 5 games,
in raw game points. Run (a) learns from clipped rewards (`reward.clip = true`, `ppo.normalize_returns = false`), run (b) from
the real points with return normalisation (`reward.clip = false`, `ppo.normalize_returns = true`).

| Env steps | (a) clipped | (b) real points, normalised returns |
|---|---|---|
| 500k | 300 | 60 |
| 1M | 210 | 210 |
| 1.5M | 240 | 110 |
| 1.8M (final) | 210 | 210 |

The final scores tie. (a) was ahead at 500k and 1.5M and never behind, and the mean score of the sampled training
games over the last log window was 702 for (a) and 647 for (b). All 5 greedy games of each evaluation scored the
same, so each number is in effect one game, and neither policy is good yet at this length of training. The rule
fixed before this run was to turn return normalisation on for Atari PPO if (b) ended level with or above (a); it ended
level, so it is on by default. That is a tie on one seed, not evidence that (b) is better.

### MinAtar

[MinAtar](https://github.com/kenjyoung/MinAtar) (Young & Tian 2019) shrinks five Atari games to a 10 x 10
grid, so every grid model kind (and its attention heatmap) works on it at Snake-like speed. `game.title`
picks Breakout, Space Invaders, Freeway, Seaquest or Asterix.

MinAtar adds "motion" channels (for example a trail where the ball was one step earlier) so that one frame
shows which way things move. `game.trails = false` removes them; then direction can only come from earlier
frames (`env.history_length` above 1, or a temporal kind). The Watch tab draws each channel as a coloured
layer with a legend.

**Measured (Breakout, PPO, defaults, 1M env steps, greedy evaluation of 30 games; not a benchmark).**
Mean score, single seeds in brackets:

| model | frames | seeds | trails on | trails off |
|---|---|---|---|---|
| `cnn` | 1 / 4 | 2 | 5.4 (1 frame) | 6.1 |
| `lstm` | 4 | 1 | 5.8 | 7.1 |
| `frame_attention` | 4 | 2 | **20.3** (19.6, 21.0) | 14.8 (14.0, 15.6) |
| `jepa_tokens` | 4 | 2 | **23.6** (32.8, 14.4) | 16.1 (15.7, 16.5) |
| `jepa_temporal`, `d_model` 64 | 2 | 1 | 10.8 | 5.0 |

- The network matters most: with the trails, `frame_attention` and `jepa_tokens` score about four times `cnn`.
  `jepa_tokens` varies a lot between seeds, so it is not clearly better than `frame_attention`.
- Removing the trails costs the strong models a quarter to a third, although the stacked frames still hold
  the direction. `cnn` and `lstm` do not use direction yet, so the trails make no difference to them.
- `jepa_temporal` had to be made lighter to run (its load comes from one token per cell per frame); at
  `d_model` 32 it was worse (6.1 and 3.6). `temporal_transformer` (400 tokens) made the machine unresponsive
  and was stopped.

### Two-player Atari (self-play)

```bash
uv run python scripts/train.py --game atari_versus --run-dir runs/atari_versus/boxing --set game.title=boxing
uv run python scripts/evaluate.py runs/atari_versus/boxing --games 20      # against a random player
```

`game.title` picks `pong` (default), `combat_tank`, `combat_plane` or `boxing` (the two-player modes of the
Arcade Learning Environment, through `multi_agent_ale_py` and the ROMs `ale-py` ships).

- **Self-play with the unchanged PPO or DQN.** A match fills two of the `env.num_envs` parallel games, one per
  player, so `num_envs` must be even. One network plays both sides and learns from both; one constant input
  plane tells it which player it is. Rewards are zero-sum.
- **Progress is measured against a random player** (seeded): self-play scores hover around zero whatever the
  model knows. Evaluations report `win_rate`, `draw_rate` and `loss_rate`; Watch shows the same match-up.
- **Not built:** a frozen earlier checkpoint as opponent, and league play.

**Measured (Boxing, PPO, `cnn`, the defaults of that time, so `reward.clip = true`, 2.27M env steps in 30 minutes; one seed, not a benchmark).** It learned
nothing: against the random player it lost 9 of 10 games at every evaluation from 0.5M steps on (mean margin
-17 to -21). The policy stayed almost uniform (entropy 2.87 of 2.89) and the value function explained none of
the return, so the greedy action was nearly arbitrary (one button on 299 of 300 steps). A random player in the
same seat won 8 of the same 10 games. A longer run, another title or other settings have not been tried.

**Follow-up with a fixed random opponent (Boxing, PPO, `cnn`, the same settings, 635k env steps; one seed, not a benchmark).**
Only the model's seat learned, and it looked the same: entropy 2.88 of 2.89, explained variance 0 at the end, and no
greedy win in 16 games at any evaluation up to 560k steps. Scripted tests of seats, rewards, observations and
rollouts found no bug. The run was short and its learning rate was already near zero, so the moving opponent is not
ruled out and the cause is still open. Single-player Boxing has not been tried as a control.

## The app

| Tab | What it is for |
|---|---|
| **New training** | Pick a game, model, algorithm, reward scheme and settings (each with a description), and start a run or add it to the queue. |
| **Runs** | Live status, score chart, log, config and history of a run, and its evaluations. Stop, resume, run an evaluation now, or **continue with changes** (a fork: a new run from a chosen checkpoint with edited rewards and training settings). |
| **Queue** | Runs waiting to start, for all games. They start by themselves, oldest first, while fewer than "runs at once" (default 1) queue-started runs train. Saved in `runs/queue.json`. See [Training queue](docs/CLI.md#training-queue-app). |
| **Reward search** | Let the reward adjust itself, judged by the real score. See [Reward search](#reward-search). |
| **Architecture** | Diagram, text tree and parameter count for a config or an existing run. |
| **Watch & attention** | Watch a checkpoint play, with the attention heatmap over the board (or bars per input number, or a coarse heatmap over an Atari screen; two-player Atari plays the model against a random player), and the action probabilities, Q-values or continuous action means. |
| **Rewards** | Create and edit named reward schemes, with a live table of what a scheme pays. |
| **Compare** | Score curves or evaluation metrics of several runs on one chart, and the config fields that differ. |
| **CLI cheat sheet** | The contents of [docs/CLI.md](docs/CLI.md). |

## Models

Chosen with `[model] kind` (or the dropdown in the app). Every kind reads grids and vectors, and
all but `resnet` and `mlp` read Atari pictures; only the input stage changes. The app marks the
recommended kinds for the chosen game.

| Kind | What it does | Attention view |
|---|---|---|
| `transformer` | One token per board cell; attention over the board. | Board heatmap |
| `cnn` | Convolutions with a linear readout. Baseline. | None |
| `resnet` | Standard two-convolution residual blocks (residual MLP blocks on vectors). | None |
| `hybrid` | A conv stem embeds each cell with its neighbourhood, then transformer blocks. | Board heatmap |
| `temporal_transformer` | Attention over the cells of the last K frames plus one token per past action. | Per frame |
| `frame_attention` | A CNN turns each of the last K frames into one vector; attention across frames. | Weight per frame |
| `lstm` | A CNN (or MLP) per frame, then an LSTM over the last K frames (its state is reset every step). | None |
| `jepa` | A world model: plays the action whose predicted next state is worth most. Experimental. | None |
| `jepa_attention` | `jepa` with `hybrid`'s attention encoder. Experimental. | Board heatmap |
| `jepa_tokens` | `jepa_attention` that predicts the next state token by token. Experimental. | Board heatmap |
| `jepa_temporal` | `jepa` with `temporal_transformer`'s encoder. Experimental. | Per frame |
| `mlp` | A plain multilayer perceptron. Baseline for vector games. | None |

- **History.** The temporal kinds (`temporal_transformer`, `jepa_temporal`, `frame_attention`, `lstm`) need
  `env.history_length` above 1 (and `env.history_actions = true`); the app sets this for you.
- **Vectors.** On Lunar Lander every number becomes a token where a board cell would. `mlp` is the
  recommended kind there; the attention kinds are for looking, not for scores.
- **`model.residual = true`** builds the conv stem of `hybrid`, the jepa attention kinds, `frame_attention`
  and `lstm` from residual blocks. **`model.layer_norm = true`** adds normalisation layers to the part that
  reads the observation. Both are off by default.
- **jepa.** An encoder turns the observation into an embedding, a predictor predicts the next state's
  embedding for every action, and a value decoder values embeddings. An optional prediction loss
  (`prediction_coef`, 0 = off by default) trains the predictor against a slowly updated (EMA) copy of the
  encoder, as in BYOL / I-JEPA. `model.rollout_steps` trains it on its own predictions over several steps.
  Details in [docs/CLI.md](docs/CLI.md#jepa-acting-on-predicted-next-states) and the docstrings.

### Measured

**Kinds on Snake 6x6 (DQN defaults, 200k env steps, greedy evaluation of 50 games; three seeds, mean with
the lowest and highest seed).**

| kind | 200k | games won (of 150) |
|---|---|---|
| `cnn` | 16.9 (16.2-17.3) | 0 |
| `jepa` | 15.5 (15.3-15.9) | 0 |
| `hybrid` | **18.9** (18.6-19.3) | 5 |
| `jepa_attention` | 18.0 (17.6-18.7) | 0 |
| `jepa_tokens` | 16.6 (16.0-17.1) | 1 |

The attention encoder helps (`hybrid` and `jepa_attention` end 1-3 apples above `cnn` and `jepa`). The jepa
predictor does not: `hybrid`, without one, is best. Three seeds are still few: "attention yes, predictor no
gain so far", not a ranking.

**The jepa prediction loss.** Trained from the start, it never improved the score in single-seed runs on
Snake and Lunar Lander and cost about 20-50% of the speed, so it is off by default. Turned on in a fork of a
trained `jepa_tokens` run (Snake 8x8, 500k more steps, `prediction_coef = 0.1`,
`prediction_trains_encoder = true`), it helped a little but consistently: **54.2** apples against 51.7 for
the same fork without it, five seeds each, every seed with the loss above every seed without (win rate 66%
against 53%). It works by shaping the shared encoder, not by planning. `value_training = "actual"` does not
learn: the predictor's error in a next state's value is as large as the differences between actions.

**Lookahead through the predictor (`jepa_tokens`, Snake 6x6, DQN, 150k env steps, 100 games; one seed, one
checkpoint).** Depth 2 picks the action with the best `reward + gamma * max Q(next state)`.

| | without rollout | `rollout_steps = 3` |
|---|---|---|
| greedy on Q(s, a) | 16.5 | 12.1 |
| depth 2, learned predictor | 16.8 (+0.3 ± 0.6) | 12.5 (+0.3 ± 0.8) |
| depth 2, real game rules (the ceiling) | 21.2 (+4.7 ± 0.6) | 22.0 (+9.8 ± 0.7) |
| error of the value read from a chained prediction | 1.56 | 0.34 |

Rollout training cut the chained value error to a fifth, but what is left is still about the size of the
gap between actions, so lookahead with the learned predictor gains nothing. A perfect model would gain
5-10 apples. Search is therefore not built.

**Hidden velocities (Lunar Lander, `hide_velocities = true`, 8 frames, PPO, 1M env steps; one seed, not a
benchmark).** Neither temporal kind learned to land (`temporal_transformer` 6%, `jepa_temporal` 2%, mostly
hovering into the time limit). `jepa_attention`, which sees the 8 frames only as extra values per token,
landed 96% of the games with a mean of 270. So attention over time did not matter here.

**Residual stem (Snake 8x8, PPO, about 1.5M env steps; one seed, not a benchmark).** `hybrid` reached 29.5
without and 23.4 with `residual = true`; `frame_attention` 24.3 and 25.2. The residual stem started slower
and was 5-8% slower per step.

## Training algorithms

`[train] algorithm` picks how a run learns; its own settings are the section of the same name.
Every model kind works with both; only the head differs.

| Algorithm | What it is |
|---|---|
| `ppo` (default) | Proximal policy optimisation: on-policy actor-critic with GAE and a clipped surrogate. Also trains continuous actions. |
| `dqn` | Double DQN: off-policy, epsilon-greedy, a replay buffer and a target network. Optional (all off by default): prioritized replay, n-step targets (`dqn.n_step`), a dueling head (`dqn.dueling`), a squared-error loss (`dqn.loss = "mse"`). |

```bash
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/dqn --set train.algorithm=dqn
```

A new DQN run takes the game's DQN defaults from `configs/<game>/algorithms/dqn.toml`. Two more switches
work with both algorithms, both off by default: `model.layer_norm` and `train.weight_decay` (AdamW above 0).
See [Choosing an algorithm](docs/CLI.md#choosing-an-algorithm).

### Continuous actions

`game.continuous = true` (Lunar Lander and Atari, off by default, locked on fork) replaces the discrete
actions by a vector of numbers: Lunar Lander's two throttles in [-1, 1], or Atari's continuous joystick.
PPO then trains a Gaussian head (a mean per number and a learned spread). The game clips the samples;
evaluation plays the mean. DQN, `env.history_actions` and the jepa kinds need discrete actions and are
refused. The Watch tab shows each number's mean with its spread.

### Measured

All one seed, not a benchmark. Greedy evaluation of 50 games.

- **PPO against DQN.** Lunar Lander, 500k env steps: DQN reached 203 (41 of 50 landed) in under 3 minutes,
  PPO -29 (it needs its full 2M steps). Snake 6x6 with `cnn`, 1M env steps: DQN 29.3 with 62% won, PPO 13.1
  with none won, but DQN was about 8 times slower per env step (four gradient steps per step of the 16 games).
  These are each algorithm's defaults, not tuned against each other.
- **DQN switches** (Lunar Lander, 500k env steps, mean score / landed at 500k): baseline 212 / 78%,
  `n_step = 3` 268 / 96%, `dueling = true` 271 / 96%, `loss = "mse"` 206 / 80%, all three 281 / 98%. Every run
  swings by 100 points or more between evaluations, so this only shows that all three train. The dueling
  head was about 30% slower.
- **Layer norm and weight decay** (Lunar Lander DQN, 500k env steps): baseline 212, `layer_norm` 141,
  `weight_decay = 0.01` 239, both 269; within the swings above. On Snake 6x6 (DQN, `cnn`, 200k) layer norm
  started slower and ended level (17.1 against 17.3). Both train; what they are meant for, keeping a network
  trainable over millions of steps, runs this short cannot show.
- **PPO normalisation** (Lunar Lander, `mlp`, 4M env steps, one seed, not a benchmark; greedy evaluation of 50 games).
  `ppo.normalize_observations`, `ppo.normalize_returns` and `ppo.clip_value_loss` all off against all on:

  | Env steps | All off: mean score / landed | All on: mean score / landed |
  |---|---|---|
  | 500k | -2 / 0% | 268 / 100% |
  | 1M | 238 / 100% | 282 / 100% |
  | 2M | 271 / 96% | 287 / 100% |
  | 4M | 263 / 96% | 290 / 100% |

  The three switches were only measured together, so this does not say which of them helped.
- **Continuous Lunar Lander** (the defaults with `game.continuous = true`, 2M env steps): 262 with 100% landed
  at 1M, 262 / 96% at 2M; the discrete run landed every game at 2M with 256. So it learns at least as fast.

## Configs, runs and evaluation

- `configs/<game>/default.toml` lists every setting with a comment. `configs/<game>/rewards/*.toml` are the
  game's named reward schemes; choosing one copies its values into the run.
- Each run lives in `runs/<game>/<name>/`: its resolved `config.toml` (the source of truth), checkpoints,
  `metrics.jsonl`, `evaluations.jsonl`, `run.json` (history of starts, resumes and forks) and `train.log`
  (when started from the app). See [What a run directory contains](docs/CLI.md#what-a-run-directory-contains).
- An **evaluation** plays complete games with the best action, on its own seed, and reports statistics per
  game (Snake: win rate, steps between apples; Lunar Lander: landing, crash, fly-off and timeout rates).
  It runs during training every `[evaluation] interval` env steps, from the Runs tab, or with
  `scripts/evaluate.py`. It does not change training, and the same checkpoint, games and seed give the same
  numbers.

## Reward search

The **Reward search** tab tests changes to a run's reward against the game itself. Starting from a
checkpoint, each generation forks the best model so far `population` times and trains each fork. One
member keeps the current reward; the others train on a bounded, seeded random change of the fields you
ticked. The member with the best real statistic (default the mean score; never the return, which depends
on the reward) wins and starts the next generation. Members are ordinary runs through the queue. Settings
are in [docs/CLI.md](docs/CLI.md#reward-search-app).

**Measured (Snake 6x6, `cnn`, PPO, 4 members x 4 generations x 250k env steps; one seed, not a benchmark).**
The comparison is a chain of four forks that keeps the parent's reward.

- **From the default reward** (searching `step`, `death`, `closer_to_food`): the winner's lineage ended at
  18.09 against 18.73 for the unchanged chain. The search found no reward better than the default.
- **From a bad step cost** (`step = -0.03`, so the parent learned to die early; searching `step` only): the
  search moved the step cost to -0.0049, next to the hand-tuned default of -0.005, and ended at 8.51 against
  6.15, with games more than twice as long. So it can repair a reward that is clearly wrong.

The winner is the best of several noisy scores on the same games, so its margin per generation is biased
upward.

## Caveats

- **Stopping the app stops the trainings it started.** Start a long run from the terminal if it should
  outlive the app. A stopped run finishes its current update and saves a checkpoint first.
- **Runs started from a terminal show as `external` in the app.** They can be watched and forked there, but
  not stopped or resumed.
- **A fork cannot change the game, algorithm, network or observation**: the game, `train.algorithm`,
  `[model]` (except the jepa fields that only shape the training loss), `history_length`, `history_actions`
  and rules such as the board size or the Atari title. The parent's weights would not fit.
- **Extending a run raises the learning rate.** The linear decay is stretched over the new total, so the rate
  jumps up at the resume point.
- **Keep dropout at 0.** PPO's probability ratio becomes noisy with dropout on.
- **DQN's replay buffer is not saved and lives in memory.** A resumed or forked DQN run starts with an empty
  buffer. Its size is about 2 x `buffer_size` x (numbers in one observation) x 4 bytes (Snake 8x8 with 100k
  transitions: about 150 MB; history frames multiply it).
- **Evaluations reproduce exactly only on the same device.** GPU and CPU can differ in the last bits and
  now and then pick a different action. With `keep_checkpoints = false` older evaluations cannot be re-checked.
- **One-seed results are anecdotes.** Most measurements above are single runs; differences of a few points
  are often within the noise.

## Development

```bash
uv run pytest -q                   # everything, about 4 minutes: run before every pull request
uv run pytest -q -m "not slow"     # about 1 minute: skips the 45 tests that train a run end to end
```

The code is organised so that each module has one job, and packages talk through small contract files
(their docstrings say what crosses them). Runs from before there were several games can be moved with
`uv run python scripts/migrate_runs.py`.
