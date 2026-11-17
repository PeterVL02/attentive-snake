# Attentive Snake

Train neural networks to play games with reinforcement learning (PPO or DQN), and look
inside them: which board cells or input numbers a transformer attends to, how the
architecture is put together, and how different models and reward schemes compare.

The project contains four games (Snake, Gymnasium's Lunar Lander, the Atari 2600 titles of Gymnasium's ALE and MinAtar), a
game-agnostic headless parallel RL environment layer, six selectable policy
architectures, two training algorithms (PPO and DQN), an architecture-diagram tool, and a local web app
that ties it all together.

## Games

| Game | Observation | What the model sees |
|---|---|---|
| `snake` (default) | grid | The whole board: one cell per position, channels head / body / food. |
| `lunar_lander` | vector | 8 numbers: x, y, vx, vy, angle, angular velocity, left and right leg contact (Gymnasium's `LunarLander-v3`; four discrete actions, or two throttles with `game.continuous = true`). Watched as the game's own picture. |
| `atari` | image | The console's screen, shrunk to 84 x 84 grayscale, the last 4 frames. One game for all titles; `[game] title` picks one (see [Atari](#atari)). |
| `minatar` | grid | MinAtar's 10 x 10 grid, one 0/1 channel per kind of object (Breakout: paddle, ball, trail, brick). One game for five titles; `game.trails = false` removes the motion channels (see [MinAtar](#minatar)). |

### Snake rules

`[game]` in a Snake config (`configs/snake/default.toml`) sets the rules; besides the board size,
`initial_length` and the starvation limit there are two optional apple rules, both off by default (the
observation keeps its shape, the food channel marks every apple):

- `food_count = N` keeps up to N apples on the board; eating one spawns one new apple on a free cell, and the
  game is won when the snake fills the board.
- `food_distribution = "center"` draws each new apple from the free cells weighted by
  `exp(-d^2 / (2 * food_spread^2))`, d being the distance to the board's centre (`food_spread` in cells,
  default 2); `"uniform"` (the default) is the classic rule and plays exactly the games it always did for a seed.

Neither rule is locked on fork, so a run trained with several or centred apples can be continued on the standard game
(a curriculum): `--fork runs/snake/PARENT --set game.food_count=1 --set game.food_distribution=uniform`. Scores under
different apple rules are not comparable (more apples is easier points). Human play takes `--food-count N` and
`--center-food`.

A single-seed check (8x8, PPO, 819,200 env steps each, greedy evaluation of 100 games, not a benchmark): the default
`transformer` learned nothing in that budget with any of the apple rules (greedy mean score 0.05 standard and centred,
0.14 with 3 apples). `jepa_attention` with 3 apples reached a greedy mean score of 36.5. Forking that run onto the standard
game (`food_count = 1`) gave 18.7 / 26.7 / 31.9 at 82k / 164k / 246k steps, where `jepa_attention` trained from scratch on
the standard game was still at 0.05 (the apple-rule curriculum got it going); the same fork from the 3-apple
`transformer` run had nothing to carry over (0.05 / 0.05 / 0.13).

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

### Atari

```bash
uv run python scripts/train.py --game atari --run-dir runs/atari/pong                       # Pong, PPO, cnn
uv run python scripts/train.py --game atari --run-dir runs/atari/venture --set game.title=venture
uv run python scripts/watch.py runs/atari/pong --scale 3
```

| `game.title` | Title | Why it is here |
|---|---|---|
| `pong` (default) | Pong | The easy one: a working setup beats the built-in opponent within a few million steps. Score from -21 to +21. |
| `ms_pacman` | Ms. Pac-Man | Dense rewards, several lives, many moving objects. |
| `montezuma_revenge` | Montezuma's Revenge | The classic sparse-reward game: plain PPO or DQN usually score 0. |
| `pitfall` | Pitfall! | Sparse rewards and penalties: staying still (score 0) beats most learned play. |
| `private_eye` | Private Eye | Sparse rewards over a large map. |
| `venture` | Venture | Sparse rewards; rooms that have to be entered before anything pays. |

**Measured (Pong, PPO, `cnn`, the defaults of `configs/atari/default.toml` with `--steps 1800000`, so the
learning rate decays to 0 over 1.8M steps; one seed, not a benchmark).** On an RTX 5080 with 8 envs the run
made about 1,250 env steps/s (5,000 frames/s; the emulators on the CPU are the limit) and took 24 minutes.
Greedy evaluation, 5 games every 200k steps:

| env steps | 0.2M - 1.0M | 1.2M | 1.4M | 1.6M | 1.8M |
|---|---|---|---|---|---|
| mean score | -21 | -20.8 | -19.8 | -19.8 | -17.4 (best -17) |
| mean game length (steps) | 764 | 855 | 1538 | 3304 | 2425 |

Until 1M steps the greedy policy kept pressing one action and lost every point in the same 764 steps, while
the sampled training games already scored about -20. Take-off began at about 1.2M steps: by 1.8M it returns
the ball in long rallies and wins 3-4 points a game (training games: -17.6). Pong is usually won (+18 or more)
after 5-10M steps, i.e. one to two hours at this speed; that run was not made. Whether 18 actions
(`full_action_space = true`) slow learning compared with Pong's own 6 cannot be told from one run.

How it works, in short (all of it is settings in `configs/atari/default.toml`, each explained in the app):

- **One game, many titles.** `atari` is a single game in the game dropdown; the title is a rule. Every title
  uses the same adapter (`games/atari/`), the same config file and, with `game.full_action_space = true` (the
  default), the same 18 actions, so a network has the same shape for every title.
- **A step is 4 screen frames** (`frame_skip`); the model sees the brighter of the last two frames at each
  pixel, shrunk to `screen_size` (84) and in grayscale. On every frame the console repeats the previous action
  with probability `sticky_actions` (0.25), which is the only randomness in a game such as Pong.
- **The reward is the sign of the points** by default (`reward.clip`), so one set of training settings fits
  titles whose points differ by orders of magnitude; `mean_score` everywhere is the real score.
- **Losing a life does not end the game.** The same environment is used for training and evaluation, and an
  evaluation's score must be the whole game's. `reward.life_lost` (0 by default) can penalise a lost life.
- **Models:** every kind except `resnet` and `mlp` reads pictures (`models/input_stages/image.py`); the
  default `cnn` with `d_model = 512` is the DQN paper's network (1.7M weights).
- **DQN's replay buffer is small**: 16000 transitions take about 3.6 GB (see the CLI cheat sheet). That is far
  below the usual one million and DQN on Atari has not been measured here.
- **Not built:** two-player titles (Combat) and continuous actions; `docs/ADDING_A_GAME.md` says what they need.


### MinAtar

[MinAtar](https://github.com/kenjyoung/MinAtar) (Young & Tian 2019) shrinks five Atari games to a 10 x 10 grid
with one 0/1 channel per kind of object, so every grid model kind (and its attention heatmap over cells) works on
it, at Snake-like speed. One game, five titles chosen by `game.title` (Breakout, Space Invaders, Freeway,
Seaquest, Asterix), each with its own minimal action set (`game.full_action_space = true` gives the paper's 6).

```bash
uv run python scripts/train.py --game minatar --run-dir runs/minatar/breakout
uv run python scripts/train.py --game minatar --run-dir runs/minatar/breakout-memory \
    --set game.trails=false --set env.history_length=4
```

**`game.trails`.** MinAtar adds "motion" channels (a trail where the ball or an enemy was one step earlier;
speed channels in Freeway; direction channels in Space Invaders) so that a single frame shows which way things
move. With `trails = false` those channels are removed from the observation, and direction can only be told from
earlier frames: history (`env.history_length` above 1) or a temporal kind. The Watch tab draws each channel as a
coloured layer with a legend.

**Measured (Breakout, PPO with the defaults of `configs/minatar/default.toml`, 1M env steps per run, greedy
evaluation over 30 games at the end; 2 seeds, not a benchmark).** About 10,000 env steps/s for `cnn`, 6,300 for
`frame_attention`; all 8 runs took 14 minutes.

| model | trails | history | seed 0 | seed 1 | mean |
|---|---|---|---|---|---|
| `cnn` | on | 1 | 5.5 | 5.3 | 5.4 |
| `cnn` | off | 1 | 4.9 | 5.4 | 5.1 |
| `cnn` | off | 4 | 6.0 | 6.1 | 6.1 |
| `frame_attention` | off | 4 | 14.0 | 15.6 | **14.8** |

- `frame_attention` with 4 frames scores more than twice as much as every `cnn` setting, on both seeds.
- But removing the trail barely hurt `cnn` on one frame (5.1 against 5.4), and giving `cnn` 4 frames helped
  only a little (6.1). So at 1M steps the small `cnn` (the paper's network: one 3 x 3 convolution of 16
  channels) is not yet limited by missing direction, and this table does not separate "memory helps" from
  "`frame_attention` is the better network here". The missing run that would separate them is
  `frame_attention` with trails on (with history 4, and with history 1).

## The app

| Tab | What it is for |
|---|---|
| **New training** | Pick a model, fill in its settings (each with a description), choose the board, reward scheme and training settings, add an optional description, and start a run. The state representation (history length, past actions) is pre-set from the chosen model. |
| **Runs** | Live status, score chart, log, config and history (descriptions, where a run was continued from) of a run. Its **evaluations** (see below): the latest one's statistics, a chart of any evaluation metric, and "Run evaluation now" for any checkpoint. Stop it, resume it with an optional new step total and description, or **continue with changes**: a new run under a new name that starts from a chosen checkpoint with edited rewards and training settings. Open any checkpoint in the watch tab. |
| **Queue** | Runs waiting to start (added with "Add to queue" in New training or in the continue-with-changes dialog), for all games. They start by themselves, oldest first, while fewer than "runs at once" (default 1) queue-started runs train; runs you start yourself do not hold the queue back. "Run now" starts an entry at once, "Remove" drops it; an entry that cannot start is kept as failed with the reason. Saved in `runs/queue.json`, so it survives a restart of the app. See [Training queue](docs/CLI.md#training-queue-app). |
| **Reward search** | Let the reward adjust itself: from a run's checkpoint, each generation trains several forks, one on the current reward and the others on bounded random changes of the fields you tick; the one with the best real score (never the reward-dependent return) wins and starts the next generation. Shows every member's reward values, score and the winners; members are ordinary runs through the queue. See [Reward search](#reward-search). |
| **Architecture** | Diagram, text tree and parameter count for the config in the form or for an existing run. |
| **Watch & attention** | Watch a checkpoint play. Board games show the attention heatmap over the board; Lunar Lander shows the game's picture with attention bars per input number; Atari shows the screen with a coarse heatmap over it (one square per token of the model). Next to the game: the action probabilities (PPO), Q-values (DQN) or, for continuous actions, the mean of each action number with its spread. |
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

Three seeds (0, 1, 2) of the same setup, to see what the single-seed tables above can carry:
Snake 6x6, DQN with its Snake defaults, 200k env steps, `prediction_coef = 0`, greedy
evaluation of 50 games every 50k env steps. Two things differ from the tables above: the step
cost is the current default (-0.005; the earlier runs had -0.01), and all 15 runs ran at once
on the CPU with one thread each, so no speeds are given. Mean score (apples) over the three
seeds, with the lowest and highest seed in brackets:

| run | 50k | 100k | 150k | 200k | games won at 200k (of 150) |
| --- | --- | --- | --- | --- | --- |
| `cnn` | 9.4 (8.7-10.0) | 10.4 (9.3-11.6) | 14.2 (13.3-14.8) | 16.9 (16.2-17.3) | 0 |
| `jepa` | 8.7 (8.4-8.9) | 11.1 (10.6-11.5) | 13.5 (13.3-13.9) | 15.5 (15.3-15.9) | 0 |
| `hybrid` | 8.5 (8.1-8.8) | 13.4 (12.3-14.0) | 16.4 (15.4-16.9) | 18.9 (18.6-19.3) | 5 |
| `jepa_attention` | 8.8 (1.9-12.5) | 13.8 (11.8-15.3) | 16.0 (14.7-18.0) | 18.0 (17.6-18.7) | 0 |
| `jepa_tokens` | 9.4 (6.1-12.0) | 13.4 (11.8-14.9) | 15.8 (14.6-17.9) | 16.6 (16.0-17.1) | 1 |

Reading: at 200k the seeds of one kind lie within about a point of each other, so differences
of two apples or more are real here and smaller ones are not. The attention encoder helps:
`hybrid` and `jepa_attention` end 1-3 apples above `cnn` and `jepa`, and their seed ranges do
not overlap those of `jepa`. The jepa predictor does not: `hybrid` (no predictor) is the best
kind, `jepa_attention` is slightly below it, `jepa` slightly below `cnn`, and `jepa_tokens` ends
level with `cnn`, below `hybrid` (ranges do not overlap), which the single-seed 19.0 above did
not show. The early lead of `jepa_attention` in the single-seed table (12.3 against 6.9 at 50k)
was seed noise: at 50k its three seeds span 1.9 to 12.5 and its mean equals `hybrid`'s. Three
seeds are still few; treat this as "attention yes, predictor no gain so far", not as a ranking.

Would looking ahead with the predictor help? Measured before building it (one seed, one
checkpoint: `jepa_tokens`, Snake 6x6, DQN, 150k env steps, `prediction_coef = 1` with
`prediction_trains_encoder = true`; 100 greedy games on the same seeds per row). "Depth 2"
picks the action with the best `reward + gamma * max Q(next state)`:

| how the action is chosen | mean score | against greedy |
| --- | --- | --- |
| greedy on Q(s, a) | 16.5 | |
| depth 2, next state from the real game rules | 21.2 | +4.7 ± 0.6 |
| depth 2, next state from the learned predictor (real reward for the first step) | 16.8 | +0.3 ± 0.6 |

Reading: one step of search is worth almost five apples with a perfect model and nothing with
the learned one, so search and a reward head are not built. The value read from a chained
prediction is off by about 1.6 (mostly a constant overestimate), while the actions that
survive the step differ by about 0.3; it picks the truly best action exactly as often as
plain Q does (73%). The predictor is only ever trained on one real step, and the value
decoder on one-step predictions, so a second chained step is outside what either has seen:
making search pay needs training on multi-step rollouts first (issue #38).

That rollout training now exists (`model.rollout_steps`, docs/CLI.md): the prediction loss feeds
the predictor its own output along the actions really taken, and under DQN also pulls the
action values read from each predicted state to those of the real state. The same measurement
on a model trained that way (`rollout_steps = 3`, otherwise identical, same seed; one seed,
one checkpoint, 100 games per row):

| | without rollout | `rollout_steps = 3` |
| --- | --- | --- |
| greedy on Q(s, a) | 16.5 | 12.1 |
| depth 2, real game rules | 21.2 (+4.7 ± 0.6) | 22.0 (+9.8 ± 0.7) |
| depth 2, learned predictor | 16.8 (+0.3 ± 0.6) | 12.5 (+0.3 ± 0.8) |
| error of the value read from a chained prediction (mean, signed) | 1.56, +1.52 | 0.34, -0.12 |
| value gap between the actions that survive the step (mean) | 0.56 | 0.53 |
| depth 2 with the predictor picks the truly best action (plain Q) | 72.6% (72.4%) | 65.5% (65.2%) |

Reading: the rollout did what it was built for, and it was not enough. The chained value error
fell to a fifth and its constant overestimate is gone, but what is left (0.34) is still about
the size of the difference between the surviving actions, and lookahead through the predictor
again gains nothing (+0.3 ± 0.8) where the real rules gain almost ten apples. The greedy
policy itself came out weaker (12.1 against 16.5); with one seed each that may be the rollout
or chance, and it is why the real rules gain more here (a weaker policy leaves more to fix). Training was about a
fifth slower (18.5 against 15.4 minutes).

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
| `ppo` (default) | Proximal policy optimisation: on-policy actor-critic. Plays rollouts with its own sampled actions, computes GAE advantages and takes a few epochs of clipped-surrogate updates on them, then throws them away. Also trains continuous actions (see [Continuous actions](#continuous-actions)). | policy (action probabilities, or a mean action and a learned spread) + value |
| `dqn` | Double DQN: off-policy and value-based. Plays epsilon-greedily, stores every transition in a replay buffer, and takes gradient steps on random batches of it with a Huber loss against a target network (the online network picks the next action, the target network values it). Optional: n-step targets, a dueling head, prioritized replay, a squared-error loss. | one Q-value per action (or a dueling value + advantage head) |

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

Three more DQN switches, each off by default (off = exactly the DQN above) and
each in the app's DQN panel:

- **`dqn.n_step`** (default 1): targets add up the next n rewards and bootstrap
  after them with `gamma ** n`. A window never crosses an episode's end and
  honours `train.bootstrap_truncated`. The replay buffer keeps single steps and
  joins them when sampling, so it costs no memory and works with prioritized
  replay; the later steps are not corrected for having been chosen by an older policy.
- **`dqn.dueling`** (default false): the head becomes a state-value stream and an
  advantage stream, `Q = V + A - mean(A)`. Works with every model kind (the jepa
  kinds: `V` of the current embedding, advantages of the predicted ones; not
  together with `value_training = "actual"`). It changes the network, so a fork must keep it.
- **`dqn.loss`** (`"huber"` or `"mse"`): how TD errors are penalised.

Measured once (Lunar Lander, DQN defaults, seed 0, 500k env steps, CPU with one
thread per run, greedy evaluation of 50 games every 100k steps; mean score /
share of games landed):

| run | 100k | 200k | 300k | 400k | 500k | time |
|---|---|---|---|---|---|---|
| baseline | -38 / 0% | -64 / 0% | 108 / 68% | 165 / 66% | 212 / 78% | 155 s |
| `n_step = 3` | -145 / 0% | 49 / 18% | 209 / 76% | 274 / 100% | 268 / 96% | 159 s |
| `dueling = true` | -24 / 0% | 181 / 54% | 187 / 70% | 173 / 70% | 271 / 96% | 213 s |
| `loss = "mse"` | 95 / 72% | -87 / 4% | 234 / 88% | 106 / 50% | 206 / 80% | 156 s |
| all three | 187 / 100% | 70 / 30% | 229 / 80% | 257 / 76% | 281 / 98% | 214 s |

One seed each, so this is an indication, not a ranking: every run swings by
100 points or more between evaluations, which is as large as the differences
between the runs. What it does show: all three switches train, none broke
learning, and the runs with 3-step targets or the dueling head ended higher
than the baseline (about 270-280 with 96-98% landed, against 212 with 78%);
3-step targets were ahead from 300k on. The squared-error loss ended level with
the baseline and was the least steady. The dueling head made a run about 30%
slower here from its first steps on (its two streams add two hidden layers to a
two-layer network; the test suite also ran beside the last two runs).

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

### Continuous actions

`game.continuous = true` (Lunar Lander and Atari, off by default, locked on fork) replaces the
discrete actions by a vector of real numbers: Lunar Lander's two throttles in [-1, 1] (main engine,
side engines; Gymnasium's own continuous mode) or Atari's continuous joystick (push 0..1, direction
-pi..pi, fire 0..1; ALE's continuous mode, turned into one of the 18 joystick actions plus a paddle
strength). PPO then trains a Gaussian head: the network outputs the mean of every number, one
spread (log standard deviation) per number is learned on its own, the same for every observation;
rollouts sample around the mean, evaluation and greedy watching play the mean. The samples are not
squashed: the game clips them to their bounds, and PPO's log-probabilities are those of the
unclipped samples (the usual PPO choice; a tanh squash would need a correction term). Every model
kind's body works with it unchanged. DQN (one value per action), `env.history_actions` (one channel
per action) and the `jepa` kinds (they predict where each of a few actions leads) need discrete
actions and are refused when the config is checked (422 in the app). The Watch tab shows the mean
of each number with its spread, drawn between the number's bounds.

```bash
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/throttle --set game.continuous=true
```

One seed, not a benchmark: Lunar Lander's defaults (`mlp`, 32 envs, 2M env steps, seed 0) with
`game.continuous = true`, greedy evaluation of 50 games every 500k env steps (`train.device = "auto"`, the GPU
here, about 7,100 env steps/s, under 5 minutes). Mean score / landed share: 221 / 90% at 508k, 262 / 100% at 1M, 263 /
98% at 1.5M, 262 / 96% at 2M (the rest crashed); training mean score about 155 at the end. The discrete
run of the [quick start](#lunar-lander-quick-start) (same defaults and seed) landed every game at 2M
with a mean score of 256, and its training score also flattened near 150. So the continuous lander
learns at least as fast here, within what one seed can show. The policy's spread did not shrink
(entropy 2.84 at the start, 2.88 at 2M, with `ppo.entropy_coef = 0.01`): the greedy mean lands, the
samples stay wide. An Atari run with a continuous joystick (Pong, `cnn`, 4,096 env steps) only proved
that it trains end to end.

### Layer norm and weight decay

Two optional switches against a network slowly losing its ability to learn over
a long run. Both are off by default, work with PPO and DQN, and are in the app's
forms; with both off, networks, checkpoints and training are exactly as before.

- **`model.layer_norm`** adds a normalisation layer after every convolution or
  fully connected layer of the part that reads the observation, and in its
  readout, each before the activation: the conv / MLP trunk of `cnn`, `resnet`,
  `mlp` and `jepa`, the stem in front of the transformer of `hybrid`,
  `jepa_attention` and `jepa_tokens`, and the per-frame encoder of
  `frame_attention` and `lstm`. On numbers it is a layer norm. On boards it is
  a group norm with one group, which normalises each board's whole feature map
  (all channels and cells) at once: the layer-norm equivalent for convolutions.
  Neither uses statistics of the batch, so playing, training batches and DQN's
  target network compute the same function. Not touched: the transformer blocks
  (they have layer norms already, so `transformer`, `temporal_transformer` and
  `jepa_temporal` do not read the switch), the jepa predictor, the LSTM itself
  and the heads. Residual blocks on numbers already have layer norms, so there
  only the readout gains one. It changes the network, so a fork must keep it.
- **`train.weight_decay`** makes every optimizer step also shrink the weights
  towards zero. 0 is plain Adam; above 0 the optimizer is AdamW (decoupled
  decay: a weight is multiplied by `1 - learning_rate x weight_decay` per
  step). Only the weight matrices of linear, convolution, attention and LSTM
  layers shrink; biases, normalisation layers and embeddings (positions, class
  token, actions) do not. A fork may switch it on, off or change it: every
  parameter keeps its Adam moments. Resume takes no training settings.

Measured once (seed 0, CPU with one thread per run, runs sharing the machine).
Lunar Lander, DQN defaults (`mlp`), 500k env steps, greedy evaluation of 50
games every 100k steps; mean score / share of games landed:

| run | 100k | 200k | 300k | 400k | 500k | time |
|---|---|---|---|---|---|---|
| baseline | -38 / 0% | -64 / 0% | 108 / 68% | 165 / 66% | 212 / 78% | 155 s |
| `layer_norm = true` | -242 / 16% | 85 / 28% | 81 / 28% | 72 / 38% | 141 / 64% | 178 s |
| `weight_decay = 0.01` | 45 / 60% | -230 / 4% | 198 / 84% | 255 / 92% | 239 / 86% | 156 s |
| both | 176 / 66% | 169 / 74% | 46 / 40% | 112 / 58% | 269 / 98% | 181 s |

Snake 6x6, DQN defaults with the `cnn` kind, 200k env steps, greedy evaluation
of 50 games every 50k steps; mean score (apples), no game won in either:

| run | 50k | 100k | 150k | 200k | time |
|---|---|---|---|---|---|
| baseline | 9.4 | 11.6 | 14.5 | 17.3 | 805 s |
| `layer_norm = true` | 1.5 | 11.0 | 13.9 | 17.1 | 857 s |

One seed each, so this is an indication, not a ranking. On Lunar Lander every
run swings by 100 points or more between evaluations, as much as the runs
differ from each other; the two runs with weight decay ended above the
baseline (239 and 269 against 212), the run with layer norm alone below it
(141), and neither difference is larger than those swings. On Snake the
layer-norm run started slower (1.5 apples at 50k against 9.4) and was level
from 100k on (17.1 against 17.3 at 200k; the three-seed range of this baseline
above is 16.2-17.3). What the runs do show: both switches train with DQN on a
vector and a grid game and neither broke learning. Weight decay cost no time;
layer norm made a run 15% slower on Lunar Lander (two extra layers in a
two-layer network) and 6% slower on Snake (tests ran beside the runs, the whole
suite beside the Snake pair, so times compare only within a table). The effect these
switches are meant for, keeping a network trainable over millions of steps, is
not something runs this short can show; weight decay 0.01 is AdamW's usual
default, not a tuned value.

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

## Reward search

The reward a model trains on is a guess about what leads to playing well. A **reward search** (the
**Reward search** tab) tests changes of that guess against the game itself, in the style of population-based
training of reward weights: starting from a run's checkpoint, each generation forks the best model so far
`population` times and trains each fork `steps_per_member` env steps. One member keeps the current reward;
the others train on a bounded, seeded random change of the fields you ticked. Every member then plays the
same greedy evaluation games, and the one with the best **real** statistic (default the mean score; never
the return, which depends on the reward being searched) wins: its model and its reward start the next
generation. Ties go to the unchanged reward. Member runs (`<search>-g<generation>-m<member>`) go through the
training queue and are ordinary runs. Settings, bounds and the handling of coefficients that start at 0 are in
[docs/CLI.md](docs/CLI.md#reward-search-app); the state is saved in `runs/reward_searches/<search>.json` and
the search goes on after a restart of the app.

**First reading (one seed, not a benchmark).** 6 x 6 Snake, `cnn`, PPO with the default settings. The parent
trained 500k env steps (greedy mean score 2.51 over 100 games). The search: 4 members x 4 generations x 250k
env steps, searching `step`, `death` and `closer_to_food` (`perturbation` 0.5, `max_change` 1.0,
`zero_scale` 0.02, seed 0), judged by the mean score over 100 greedy games. The comparison is the lineage that
keeps the default reward: a chain of four forks of 250k steps each from the same parent checkpoint (the same
total steps and the same learning-rate schedule per fork). Whole measurement: under 7 minutes of compute.

| Generation | Winner's step / death / closer_to_food | Winner's mean score | Unchanged member (same generation) | Unchanged chain (same total steps) |
|---|---|---|---|---|
| 0 | -0.005 / -1.0 / 0 (unchanged won) | 5.72 | 5.72 | 4.79 |
| 1 | -0.0056 / -1.20 / 0.0001 | 9.67 | 8.09 | 8.71 |
| 2 | -0.0057 / -1.16 / -0.0075 | 14.39 | 13.75 | 13.56 |
| 3 | -0.0059 / -1.23 / -0.0035 | 18.09 | 17.98 | 18.73 |

What it shows: the search runs end to end through the app and the queue (16 members, one at a time, none failed),
the reward drifts within its bounds (death a little more negative, the step cost a little larger), and after
1M steps the winner's lineage scores 18.09 against 18.73 for the lineage that kept the default reward. So on
this setup the search found **no reward better than the default**; the difference is well within what
another seed would change. What it does not show: anything about larger boards, other games or longer
searches. Also note that the per-generation margins over the unchanged member are biased upward: the winner is
the best of four noisy scores on the same 100 evaluation games, and the next generation does not re-check it on
fresh games.

**Second reading: repairing a bad step cost (one seed, not a benchmark).** Same setup, but the parent trained
500k steps with `step = -0.03`, so that 100 steps of living cost three times as much as dying: it learned to
die early (mean score 1.56, games of 8 steps). The search changed only `step` (4 members x 4 generations x 250k
steps, `perturbation` 0.5, `max_change` 1.0, seed 0); the comparison is a chain of four 250k forks that keep
`step = -0.03`. Whole measurement: 6 minutes.

| Generation | Winner's step | Winner's mean score (game length) | Unchanged chain (game length) |
|---|---|---|---|
| 0 | -0.0183 | 2.74 (14) | 2.58 (13) |
| 1 | -0.0183 (unchanged won) | 4.20 (25) | 3.32 (18) |
| 2 | -0.0183 (unchanged won) | 5.66 (40) | 4.32 (27) |
| 3 | -0.0049 | 8.51 (90) | 6.15 (38) |

The search moved the step cost from -0.03 to -0.0049, next to the -0.005 that was later set by hand as the
default, and its lineage ends at 8.51 against 6.15 for the bad reward, with games more than twice as long.
The same upward bias of the winner applies, but the lineage was ahead in every generation and the unchanged
member of the last generation (7.58, on -0.0183) already beat the chain. So the search can repair a reward
that is clearly wrong; on the hand-tuned default (first reading) it found nothing to improve.

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
  `train.weight_decay` may change on a fork (on, off, another value; the moments
  are kept); `model.layer_norm` may not (it changes the network).
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
uv run pytest -q                   # everything, about 4 minutes: run before every pull request
uv run pytest -q -m "not slow"     # about 1 minute: skips the 45 tests that train a run end to end
```

The code is organised so that each module has one job; see [CLAUDE.md](CLAUDE.md)
for a map of the packages and the contracts between them.
