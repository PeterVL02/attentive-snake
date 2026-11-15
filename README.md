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
| `resnet` | A conv stem, then residual blocks (two convolutions and a skip each) and a linear readout; residual MLP blocks on vectors. | None |
| `hybrid` | A conv stem embeds each cell with its neighbourhood, then transformer blocks. | Board heatmap |
| `temporal_transformer` | Attention over the cells of the last K frames plus one token per past action. | Board heatmap per frame |
| `frame_attention` | A CNN turns each of the last K frames into one vector; attention runs across frames. | Weight per frame |
| `lstm` | A CNN (or MLP) turns each of the last K frames into one vector and an LSTM reads them oldest to newest. | None |
| `jepa` | A world model: an encoder, a predictor of each action's next-state embedding, a value decoder; plays the action whose predicted next state is worth most. Experimental. | None |
| `jepa_attention` | `jepa` with `hybrid`'s attention encoder: the class token's output is the embedding the predictor works on. Experimental. | Board heatmap |
| `jepa_tokens` | `jepa_attention` with a per-cell predictor: the embedding is every token (class token + one per cell), a small transformer predicts the next state's tokens cell by cell, the decoders read the class token. Experimental. | Board heatmap |
| `jepa_temporal` | `jepa` with `temporal_transformer`'s encoder: attention over the cells (numbers) of the last K frames plus one token per past action, the class token's output is the embedding. For games where one frame hides part of the state. Experimental. | Board heatmap per frame |
| `mlp` | A plain multilayer perceptron on the flattened observation. Baseline for vector games. | None |

Every kind reads both grids and vectors; only the input stage changes. On a
vector observation (Lunar Lander) every number becomes a token where a board cell
would, the convolutions become fully connected layers (a convolution over an
arbitrary order of numbers means nothing), and the attention view is one weight per
input number instead of a board heatmap. `mlp`
is the recommended kind for vector games; the app marks the recommended kinds
for the chosen game.

`lstm` is a DRQN-style policy with a fixed unroll length, not a recurrent agent that
remembers a whole episode: the last K frames are encoded one by one and an LSTM reads them
oldest to newest, but its hidden state is reset at every step, never carried from one env
step to the next and never stored in the DQN replay buffer. Its memory is exactly
`history_length`, like the temporal kinds. `resnet` differs from `cnn` (whose layers after the first
are already single-convolution residual layers) in using the standard two-convolution residual block
(`conv_layers` of them; on vectors, layer-normalised residual MLP blocks as in SimBa).

`hybrid` (and `jepa_attention` and `jepa_tokens`, which use its trunk), `frame_attention` and `lstm` have a convolutional stem (an MLP on numbers) in front of
their attention or recurrent part. `[model] residual = true` builds that stem (for the other two:
the encoder of each frame) from the same residual blocks as `resnet` instead of plain layers, so
you get the kind you chose with a deeper-trainable stem rather than a new kind. The other kinds
ignore it, and it is part of the network, so a continued run cannot change it. Off by default:
a network built with `residual = false` is exactly the one before the switch existed.

Single-seed check, not a benchmark: Snake 8x8 (the repo's defaults, PPO, device as in the
config, four runs sharing one machine), greedy evaluation of 100 games every 250k env steps,
stopped by a time cap at about 1.5M of the planned 2M steps (so the learning-rate decay was
not finished). Mean score (apples):

| run | 250k | 500k | 750k | 1M | 1.25M | 1.5M | env steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hybrid` | 9.4 | 16.6 | 19.4 | 22.2 | 24.2 | 29.5 | 2160 |
| `hybrid`, `residual = true` | 0.1 | 3.2 | 11.7 | 16.6 | 21.6 | 23.4 | 2020 |
| `frame_attention` (`history_length = 4`) | 3.9 | 11.4 | 17.8 | 23.1 | 25.4 | 24.3 | 2130 |
| `frame_attention`, `residual = true` | 1.4 | 7.0 | 13.8 | 21.3 | 23.6 | 25.2 | 1970 |

Here the residual stem started slower and had not caught up after 1.5M steps (`frame_attention`
nearly had), at about 5-8% lower speed; one seed cannot say whether it ever helps.

`jepa` is a JEPA-like world model, trained from scratch on the game like every other kind
(no pretrained weights, nothing downloaded). An encoder turns the observation into an
embedding (the `cnn` encoder: convolutions on a board, an MLP on numbers); a predictor takes
the embedding and an action and predicts the embedding of the next state, for every action
at once; a value decoder turns an embedding into a value. With DQN the Q-value of an action
is the decoded value of the embedding it is predicted to lead to, so the agent plays the
action whose predicted next state is worth most. With PPO the policy scores the predicted
embeddings with its own decoder and the value decoder values the current embedding. A
prediction loss (`prediction_coef`; 0 = off, the default) makes the embedding predicted for the action
taken match the embedding of the state that really followed, given by a slowly updated
(EMA, `target_ema`) copy of the encoder that gets no gradients, as in BYOL / I-JEPA; that
copy is what stops the trivial answer of one embedding for every state, and the Runs chart
shows `embedding spread` to check. With the loss on it trains the predictor only by default
(`prediction_trains_encoder = false`): the encoder learns from the reinforcement-learning loss
alone, as without the loss. `value_training` (DQN) chooses what the TD loss trains:
the decoded *predicted* embedding of the action taken (`predicted`, the default), or the
decoded embedding of the *actual* next state, i.e. the value of arriving there with the
reward for getting there (`actual`; the predictor then learns only from the prediction loss,
which also makes the decoded prediction match the decoded real next state). The Runs chart's
`value gap` is the difference between those two values.

Why `actual` does not learn (measured on Lunar Lander): it chooses an action by comparing
the values of the next states the predictor foresees, and one step changes a state very
little, so those values differ by about 0.006 between the two best actions. The predictor's
error in exactly that value was the same size (about 0.006), even though its embeddings
were close to the real ones: matching all numbers of an embedding does not make the one
direction the value reads accurate. Replaying the same states with every action showed its
choice agreed with the best real next state only about half the time, and the agent acted
close to randomly (it crashed early like a random player). Training that value directly
halved the error, but the gaps between actions shrank with it, and it still never learned
to land. `predicted` avoids the problem because the TD loss trains each action's value
itself. The other suspects were measured and ruled out as the main cause: the EMA and the
online encoder hardly differ (copying the encoder every step changed nothing), the
embeddings did not shrink, and the values at predicted embeddings showed no upward bias.

What is JEPA about it: prediction happens in embedding space, never in pixels or numbers;
the target embedding comes from an EMA target encoder; nothing is reconstructed. What is not:
there is no masking and no large-scale self-supervised pre-training: the encoder is trained
at the same time as, and also by, the reinforcement-learning loss; the predictor looks one
step ahead only (multi-step lookahead over predicted embeddings, DFS/BFS, is not built yet,
though the predictor can be applied to its own predictions). Caveats: the predictor is
deterministic, so where the next state is random (where Snake's next apple appears) it can
only predict something like an average embedding; with `actual`, a reward that depends on
where the snake came from (Snake's closer/farther shaping) is not a function of the next
state alone, so the decoder cannot learn it exactly; and the extra encoder and target-encoder
passes make a gradient step slower than `cnn`/`mlp`.

`jepa_attention` is the same world model with `hybrid`'s encoder: the conv stem (an MLP per
number on vectors), transformer blocks over the cells and the class token's output as the
embedding the predictor and the value decoder work on. Its `num_layers` sets both the number
of transformer blocks and the predictor's depth (no separate setting), the Watch tab shows the
encoder's attention as for `hybrid`, and everything above (heads, `prediction_coef`,
`value_training`, ...) applies unchanged; the jepa part is shared code, not a copy.

Single-seed check, not a benchmark: Snake 6x6, DQN with its Snake defaults (`--set
train.algorithm=dqn`, default `[model]` sizes: `d_model` 64, 3 layers, 64 conv channels x 3),
200k env steps, `prediction_coef = 0` unless noted, seed 0, greedy evaluation of 50 games every
50k env steps, GPU, all runs sharing the machine (so speeds compare only roughly).
Mean score (apples); win rate at 200k:

| run | parameters | 50k | 100k | 150k | 200k | won | env steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cnn` | 223k | 8.5 | 11.7 | 14.2 | 16.0 | 0% | 355 |
| `jepa` | 244k | 8.0 | 11.0 | 13.9 | 16.6 | 0% | 300 |
| `hybrid` | 231k | 6.9 | 14.1 | 16.2 | 19.5 | 2% | 237 |
| `jepa_attention` | 252k | 12.3 | 14.8 | 18.0 | 19.8 | 2% | 219 |
| `jepa_attention`, `prediction_coef = 1` | 252k | 12.5 | 14.3 | 18.3 | stopped at 160k by the time cap | | about 170 |

(The `prediction_coef = 1` run was stopped at 60k to make room and resumed from its 50k
checkpoint, which refills the replay buffer.) Reading: `jepa_attention` was ahead of `jepa` at
every point (by 3-4 apples) and ended level with `hybrid`, which has the same encoder without
the predictor; it was well ahead of `hybrid` early (12.3 against 6.9 at 50k) and about 8% slower.
So most of the gain over `jepa` is the attention encoder; what the predictor adds on top of it
was faster early learning in this one run, not a better end result. The prediction loss again
changed nothing measurable and cost about 30% of the speed. `cnn` and `jepa` were level, as before.

`jepa_tokens` keeps every token of that encoder: the state embedding is the class token plus
one token per cell (per number on vectors), and the predictor is a small transformer
(`predictor_layers` blocks, default 1; `num_layers` is the encoder's) that adds the action's
embedding to every token and predicts the next state's whole token set, residually
(`tokens + f(tokens, a)`). A Snake step changes two or three cells, so the predictor mostly
has to copy. The decoders read the class token of a token set, so the heads are those of the
other two jepa kinds; the prediction loss compares the predicted and target token sets token
by token (cosine distance per token, averaged). The predictor runs once per action.

Single-seed check, same setup as the table above (those rows are from the earlier run and
were not repeated); the three `jepa_tokens` runs shared the GPU with each other (and with
another agent's work), so speeds compare only roughly. Mean score (apples); win rate at 200k:

| run | parameters | 50k | 100k | 150k | 200k | won | env steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `jepa_attention` (earlier run) | 252k | 12.3 | 14.8 | 18.0 | 19.8 | 2% | 219 |
| `hybrid` (earlier run) | 231k | 6.9 | 14.1 | 16.2 | 19.5 | 2% | 237 |
| `jepa_tokens`, `prediction_coef = 0` | 285k | 11.1 | 14.9 | 16.1 | 19.0 | 4% | about 190 |
| `jepa_tokens`, `prediction_coef = 1` (predictor only) | 285k | 12.1 | 15.8 | 16.5 | 18.1 | 0% | 115 (3 runs) to 170 (2 runs) |
| `jepa_tokens`, `prediction_coef = 1`, loss also trains the encoder | 285k | 11.4 | 14.7 | 17.1 | stopped at 180k by the time cap | | 97 (3 runs) to 155 (2 runs) |

The prediction metrics (training log, 20k / 100k / 200k env steps; the third run's last is 180k):

| run | `prediction_loss` | `embedding_std` | `value_gap` |
| --- | --- | --- | --- |
| predictor only | 0.19 / 0.13 / 0.14 | 0.080 / 0.086 / 0.091 | 0.89 / 1.56 / 2.59 |
| also trains the encoder | 0.020 / 0.009 / 0.006 | 0.022 / 0.060 / 0.062 | 1.16 / 1.64 / 2.36 |

Reading: `jepa_tokens` was level with `jepa_attention` and `hybrid` (within a point or so at
every evaluation, which is about the noise of 50 games and one seed), not better; it has 13%
more parameters (the predictor) and was somewhat slower. The prediction loss again changed
nothing measurable in the score and cost 35-50% of the speed. With the loss on the predictor
only, the per-token cosine distance stalled at about 0.14 after 50k steps although most cells
do not change: the encoder, trained by the TD loss alone, keeps moving its tokens, and the
target follows it. Letting the loss train the encoder made the tokens 20 times easier to
predict (0.006) with somewhat less spread (0.06 against 0.09; no collapse), and still did not
change the score. The value gap grows in both (the value decoder is trained on predictions
only with `value_training = "predicted"`, so that is expected and harmless).

Lunar Lander can hide part of what the model sees: `game.hide_velocities = true` replaces
the three velocity numbers by zeros (the physics, score and reward still use the true
state), so a model has to infer them from several frames and memory can matter. The rule
is locked when forking a run.

The temporal kinds (`temporal_transformer`, `jepa_temporal`, `frame_attention`, `lstm`) need history: set `history_length` (for example 4) and
`history_actions = true` under `[env]`. The app does this for you.

`jepa_temporal` is the jepa world model (above) with `temporal_transformer`'s encoder, made
for exactly this: one token per number (cell) of each of the last K frames plus one per past
action, transformer blocks, and the class token's output as the embedding `z` the predictor and
the decoders work on, so `z` can hold the velocities a single frame no longer shows. Its
`num_layers` sets both the transformer blocks and the predictor's depth (as for
`jepa_attention`); attention is shown per frame as for `temporal_transformer`. The token
construction is `temporal_transformer`'s own code (`models/policies/temporal_tokens.py`), not
a copy.

Single-seed check, not a benchmark: Lunar Lander with `game.hide_velocities = true`, PPO with
the game's defaults (32 envs x 256 steps, default `[model]` sizes: `d_model` 64, 2 layers, 4
heads), `env.history_length = 8`, `env.history_actions = true`, 1M env steps, seed 0, greedy
evaluation of 50 games every 500k env steps (it runs at the first update past each multiple:
508k and 999k), GPU, the four runs sharing the machine (about 10 minutes together, so speeds
compare only roughly). Mean score / landing rate:

| run | parameters | 500k | 1M | at 1M: crashed / flew off / timeout | training mean score at 1M | env steps/s |
| --- | --- | --- | --- | --- | --- | --- |
| `temporal_transformer` | 102k | -103 / 2% | -136 / 6% | 14% / 0% / 80% | 6 | 2070 |
| `jepa_temporal`, `prediction_coef = 0` | 119k | -185 / 0% | -111 / 2% | 38% / 0% / 60% | 23 | 1920 |
| `jepa_temporal`, `prediction_coef = 1` | 119k | -165 / 0% | -190 / 0% | 30% / 12% / 58% | 4 | 1430 |
| `jepa_attention` (history as extra values per token) | 127k | -47 / 2% | 270 / 96% | 0% / 0% / 4% | 274 | 1740 |

With the loss on, `prediction_loss` (cosine distance) was 0.003 at 41k, 0.019 at 205k, 0.014 at
410k, 0.023 at 614k, 0.028 at 819k and 0.023 at 1M; `embedding_std` rose from 0.017 to about
0.10 (1/sqrt(64) = 0.125, no collapse) and `value_gap` fell from 0.23 to about 0.06.

Reading: neither temporal kind learned to land here. The earlier single-seed result (only
`temporal_transformer` with 8 frames landed, 86%, mean 213, before several changes to the
code) was not reproduced: its network is unchanged (its weights at seed 0 and its outputs are
pinned by a test), so that was luck of the seed or of other changes, not investigated.
`jepa_temporal` was level with `temporal_transformer` (both hovered into the time limit; the
differences are within what one seed and 50 games can show). The prediction loss did not help:
it was the worst of the four and cost about 25% of the speed. The surprise is
`jepa_attention`, which sees the 8 frames only as extra values of each number's token (no
attention over time): it landed in 96% of the games with a mean of 270. So in this run,
attention over time did not matter for jepa; one seed each, so take it as a hint, not a result.

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
`buffer_size`. Optional **prioritized replay** (`dqn.prioritized`, off by
default) draws transitions the network predicts worst more often, as in
Schaul et al. (2016), and weights their loss to undo the bias (the weight
exponent beta rises to 1 over training). It costs some speed per step, its
priorities start fresh on resume or fork (like the buffer), and one seed on
Lunar Lander is no evidence that it helps. One DQN iteration is one step of every parallel game, so
`num_envs` only changes how experience is gathered, not how much is learned
per step.

In one test (one seed each, CPU, 500k env steps on Lunar Lander with the default
`mlp`): DQN with its defaults reached a greedy evaluation mean score of 203 (41
of 50 games landed) in under 3 minutes; PPO with its defaults and the same
budget ended at -29 (it needs its full 2M steps). On a 6x6 Snake board with
the `cnn` kind and 1M env steps, DQN with its defaults reached a greedy mean
score of 29.3 with 62 % of 50 games won; PPO with its defaults and the same
budget reached 13.1 with none won. DQN was much slower per env step there,
though (about 600 steps/s against about 5,000 on the same CPU: 27 minutes
against 3), because every step of the 16 games is followed by four gradient
steps. See the Caveats for what this does and does not say.

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
- **`jepa`'s prediction loss does not help yet, so it is off by default.** One seed each
  (seed 0), DQN, greedy evaluation of 50 games every 50k env steps; not a benchmark, nothing tuned.

  Lunar Lander, 300k env steps (mean score; landed share at the end):

  | run | 100k | 150k | 200k | 250k | 300k | landed | env steps/s |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `mlp` | -50 | -59 | -20 | 18 | 167 | 74% | 2140 |
  | `jepa`, `prediction_coef = 0` | -52 | 13 | 231 | 227 | 223 | 86% | 1060 |
  | `jepa`, `prediction_coef = 1` (`predicted`, loss trains the predictor only) | -96 | -133 | -27 | 240 | 208 | 48% (50% hovered to the time limit) | 870 |
  | the `prediction_coef = 1` run with `dqn.prioritized = true` | -119 | -324 | -308 | -140 | -198 | 0% | 820 |
  | `jepa`, `prediction_coef = 1`, loss also trains the encoder (the old default) | -65 | -93 | -108 | -19 | -187 | 6% (88% flew off) | 890 |
  | `jepa`, `prediction_coef = 1`, `actual` | -21 | -69 | -31 | -134 | -178 | 0% | 960 |

  Snake 6x6, 200k env steps planned (mean apples): `jepa` with `prediction_coef = 0` 11.3 /
  13.1 / 16.1 at 100k / 150k / 200k (277 env steps/s); `jepa` with `prediction_coef = 1` 9.7 / 14.3 at 100k /
  150k (184 steps/s; stopped at 160k by the time cap); `jepa` `actual` (`prediction_coef = 1`) 0.55 / 0.61 (stopped
  at 140k). The runs shared the machine, so the speeds compare only roughly.

  Reading: letting the prediction loss train the encoder is what hurt `predicted` (its
  gradient into the encoder was as large as the TD loss's); without that it is level with
  `prediction_coef = 0` on both games within what one seed can show, but not better, and
  it costs speed. `prediction_coef = 0` is still the fastest and at least as good. The one
  prioritized-replay run was unstable (its value gap grew from 0.02 to 0.4, some games
  scored below -1000); one seed cannot say whether that is prioritized replay with this
  kind or chance. `actual` still does not learn (see the jepa section for why).
- **History adds no information in principle.** Snake is fully observable from
  one frame, so the temporal models are for studying attention over time, not
  for better scores, and they train slower.
- **A model is tied to its board size.** A checkpoint cannot be used on a
  different grid.
- **A continued run (fork) cannot change the game, algorithm, network, board or observation.**
  The game, `train.algorithm`, all of `[model]` (except jepa's `prediction_coef`, `value_training`, `prediction_trains_encoder` and
  `target_ema`, which change how it is trained, not the network), the board size, `initial_length`, `history_length` and
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
