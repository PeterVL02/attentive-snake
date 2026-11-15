# Command-line cheat sheet

All commands run from the project root with `uv run ...`.

Every run belongs to one game: `snake` (the default) or `lunar_lander` (Gymnasium's LunarLander-v3). Each game
has its own default config (`configs/<game>/default.toml`), its own reward schemes (`configs/<game>/rewards/`) and
its own runs directory (`runs/<game>/`). `--game NAME` picks the game for a new run. Resuming, forking and watching read the game from the run's
`config.toml`.

Every run is trained with one algorithm, `train.algorithm`: `ppo` (the default; its settings are the `[ppo]`
section) or `dqn` (the `[dqn]` section). `configs/<game>/default.toml` holds the PPO defaults;
`configs/<game>/algorithms/dqn.toml` holds the game's DQN defaults, used when a new run asks for DQN without a
`--config` file.

## Play Snake yourself

```bash
uv run python -m attentive_snake.games.snake.play --width 10 --height 10 --fps 8 --cell-size 30
```

Arrow keys or WASD steer, `R` restarts after a game over, `Esc` quits. `--food-count 3` puts several apples on the
board and `--center-food` makes them appear near the middle.

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

# Snake rules: three apples at once, new apples near the middle of the board
uv run python scripts/train.py --run-dir runs/snake/3apples --set game.food_count=3 --set game.food_distribution=center --set game.food_spread=2

# Any single setting, a description, and keeping only the latest checkpoint
uv run python scripts/train.py --run-dir runs/snake/lr-test --set train.learning_rate=1e-4 --set reward.step=0 \
    --description "lower LR, no living cost" --no-keep-checkpoints
```

New and forked runs print the game's warnings about their config to stderr before training (never a prompt, so scripts
are not blocked). Snake warns when the step cost times the discounted horizon, `|step| / (1 - gamma)`, is at least
the death penalty (the default step cost of -0.005 stays below it; -0.01 with gamma 0.99 and death -1 reaches it): early on the snake may run into a wall because nothing rewards staying
alive. The app shows the same warning in a dialog before starting.

```bash
# Lunar Lander with its defaults (mlp, 32 envs, 2M steps: a few minutes on a CPU), run saved in runs/lunar_lander/mlp
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/mlp

# Another model kind; the temporal kinds want history
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/tt --set model.kind=temporal_transformer \
    --set env.history_length=4 --set env.history_actions=true

# Treat the time limit like a real ending, as runs made before train.bootstrap_truncated existed did
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/no-bootstrap --set train.bootstrap_truncated=false
```

```bash
# DQN instead of PPO: the game's DQN defaults (configs/<game>/algorithms/dqn.toml), then any [dqn] setting
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/dqn --set train.algorithm=dqn
uv run python scripts/train.py --run-dir runs/snake/dqn-cnn --set train.algorithm=dqn --set model.kind=cnn \
    --set dqn.batch_size=128 --set dqn.buffer_size=200000
# DQN with prioritized experience replay (off by default; the priority_* settings apply only when it is on)
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/per --set train.algorithm=dqn \
    --set dqn.prioritized=true --set dqn.priority_alpha=0.6 --set dqn.priority_beta_start=0.4
# DQN with 3-step targets, a dueling head and a squared-error loss (each is its own switch; all off by default)
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/dqn-3step --set train.algorithm=dqn \
    --set dqn.n_step=3 --set dqn.dueling=true --set dqn.loss=mse
# Normalisation layers in the network and weight decay in the optimizer (two separate switches, both off by default; PPO or DQN)
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/dqn-norm-decay --set train.algorithm=dqn \
    --set model.layer_norm=true --set train.weight_decay=0.01
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
- `--set train.algorithm=dqn` switches the algorithm: without `--config` the run starts from the game's DQN
  defaults; with `--config FILE` the file's `[ppo]` section is dropped and `[dqn]` takes its code defaults. Then
  `--set dqn.<key>=...` changes DQN settings. A config holding the section of another algorithm than its
  `train.algorithm` is refused.
- `--description TEXT` is saved in the run's `run.json` (see below) and shown in the app.
- `train.bootstrap_truncated` decides what a game cut short by a limit (Lunar Lander's time limit, Snake's
  starvation limit) is worth after the cut: `true` counts the value the network expects from the cut-off state,
  `false` counts nothing, like a real ending. Default: `true` for Lunar Lander, `false` for Snake (starving is one
  of its rules, with its own reward); a config without the key (every run made before it existed) is `false`. It
  can be changed when continuing a run under a new name (fork), not with `--resume`. Logged scores and returns are
  the game's own in both cases.
- `train.weight_decay` (default 0, both algorithms) makes every optimizer step also shrink the network's weights a
  little towards zero. 0 is plain Adam, exactly as before the setting existed; above 0 the optimizer is AdamW
  (decoupled weight decay: a weight is multiplied by `1 - learning_rate x weight_decay` each step; 0.0001 to 0.01
  are usual values). Only the weight matrices of the linear, convolution, attention and LSTM layers shrink; biases,
  normalisation layers and embeddings (positions, class token, actions) do not. It can be changed (switched on, off
  or to another value) when continuing a run under a new name (fork): the optimizer's moment estimates are carried
  over parameter by parameter. Not with `--resume`, which takes no training settings.
- `--no-keep-checkpoints` deletes the older checkpoints of the run each time a new one is saved, so only the
  latest remains (saved as `train.keep_checkpoints = false`). Deleted checkpoints can no longer be watched,
  continued from or re-evaluated. `--keep-checkpoints` (the default) keeps them all.
- `--eval-interval N`, `--eval-games X`, `--eval-seed S` set `evaluation.interval`, `evaluation.games` and
  `evaluation.seed` (see "Evaluate a trained agent" below). They work with new, resumed and forked runs and are saved
  in the run's `config.toml`; `--eval-interval 0` turns evaluation off.

## Resume or extend a run

```bash
# Continue from the latest checkpoint up to the run's total_env_steps
uv run python scripts/train.py --resume runs/snake/8x8

# Train longer: raise the budget (the saved config.toml is updated too)
uv run python scripts/train.py --resume runs/snake/8x8 --steps 10000000

# With a note, and from now on keep only the latest checkpoint (saved to config.toml)
uv run python scripts/train.py --resume runs/snake/8x8 --description "more steps" --no-keep-checkpoints

# Evaluate from now on every 250k env steps with 100 games (e.g. a run started before evaluation existed)
uv run python scripts/train.py --resume runs/snake/8x8 --steps 10000000 --eval-interval 250000 --eval-games 100
```

Resuming a DQN run restores its online and target networks and its optimizer, but not its replay buffer
(checkpoints do not hold it): the buffer refills from scratch, and gradient steps wait until it holds
`dqn.learning_starts` transitions again. Epsilon continues where the step count is. With `dqn.prioritized`
the priorities live in the buffer too, so they start fresh and beta (see below) continues from the step count.

`--resume` uses the run's own `config.toml` (including its game); `--game`, `--config`, `--set`, `--width`,
`--height`, `--num-envs`, `--seed` and `--fork` are rejected with it. `--steps`, `--[no-]keep-checkpoints`,
`--description` and `--eval-interval/--eval-games/--eval-seed` are accepted. A resumed run appends to its
`evaluations.jsonl`. To continue with other settings, start a new run from a checkpoint (next section).

## Continue a run under a new name with changed settings (fork)

```bash
# New run runs/snake/8x8-lowlr starting from the latest checkpoint of runs/snake/8x8, with a lower learning rate
uv run python scripts/train.py --run-dir runs/snake/8x8-lowlr --fork runs/snake/8x8 --set train.learning_rate=1e-4

# From a specific checkpoint, with other rewards, a new budget and a note
uv run python scripts/train.py --run-dir runs/snake/8x8-food2 --fork runs/snake/8x8 --fork-checkpoint checkpoint_0000500000.pt \
    --set reward.food=2 --steps 2000000 --description "food worth more"

# Back to the standard game after training with several apples (a curriculum; the apple rules are not locked)
uv run python scripts/train.py --run-dir runs/snake/8x8-single --fork runs/snake/8x8-3apples --set game.food_count=1

# From an edited copy of the parent's config
cp runs/snake/8x8/config.toml configs/8x8-variant.toml   # edit it, then:
uv run python scripts/train.py --run-dir runs/snake/8x8-variant --fork runs/snake/8x8 --config configs/8x8-variant.toml
```

- The new run loads the parent's weights and optimizer state (for DQN also the target network), then counts its own
  env steps from 0 up to its own `total_env_steps` (by default the parent's), with its own learning-rate schedule
  starting from `learning_rate`. Adam's moment estimates are carried over even when the learning rate changes (they
  do not depend on it), and also when `train.weight_decay` is switched on, off or changed (the new run's optimizer
  is built from its own settings and every parameter gets its own moments back). A DQN fork starts with an empty replay buffer and its exploration starts again from
  `dqn.epsilon_start` (lower it with `--set dqn.epsilon_start=0.1` to explore less).
- Its settings are the parent's `config.toml` (or `--config`) plus `--set`, `--steps`, `--num-envs`, `--seed`,
  `--keep-checkpoints`. Everything under `[reward]`, `[train]` (but `train.algorithm`) and the algorithm's section
  (`[ppo]` / `[dqn]`), `env.num_envs` and the game rules the game does not lock (Snake:
  `game.max_steps_without_food`, `game.food_count`, `game.food_distribution`, `game.food_spread`) may change. The game (`game.name`), the algorithm (`train.algorithm`: the weights
  end in its head), all of `[model]` except jepa's `prediction_coef`, `value_training`, `prediction_trains_encoder`, `target_ema`, `rollout_steps` and `rollout_value_coef` (they
  change how it is trained, not the network), `env.history_length`, `env.history_actions` and the rules the game locks (Snake: `game.width`, `game.height`,
  `game.initial_length`) must equal the parent's (the weights would not fit, or the input would change meaning);
  a change is rejected naming the field, before anything is written.
- `--run-dir` is required and must be a new run (a directory with checkpoints is refused). `--fork-checkpoint`
  takes a checkpoint file name of the parent (or its path); the default is the parent's latest.
- The parent is never modified. The new run's `run.json` records the parent run, checkpoint and its env steps.
  The new run starts its own `evaluations.jsonl`; the parent's evaluations are not copied.

## Training queue (app)

The queue lets runs start by themselves, one after another. It lives in the app: "Add to queue" next to
"Start training" in New training and next to "Start new run" in the dialog for a new run from a checkpoint,
and the **Queue** tab. A queued run gets the same checks as a start, at the moment it is queued: an invalid
setting, a fork that would change a locked setting, or a name that is taken (an existing run, or another
queued entry of the same game) is refused right away, and nothing is written to `runs/` until it starts.

- **When an entry starts:** whenever fewer than "runs at once" (`max_parallel`, default 1) runs started by the
  queue are training, the oldest waiting entry starts. Runs you start yourself, and entries started with
  **Run now**, take no place: they never hold the queue back. 0 pauses the queue (only Run now starts entries).
- **Run now** starts an entry immediately, whatever is training. **Remove** takes it out of the queue.
- **An entry that cannot start** (say the fork's parent checkpoint was deleted while it waited) stays in the
  list as failed, with the reason, and the next entry starts. "Try again" retries it; it no longer holds its name.
- **Once started** an entry leaves the queue and is an ordinary run: Runs tab, log, run.json, Stop.
- **One queue for all games** (one machine trains them all); each entry shows its game.
- **Saved** in `runs/queue.json` (the runs directory's root), so it survives a restart of the app. Training jobs
  still stop with the app, as before; the queue goes on with the next entry when the app starts again.

There is no command-line queue: from the terminal, run `scripts/train.py` commands one after another. A
Python script inside the app's process can use the same queue (`app.state.launcher`, see
`src/attentive_snake/app/training_launcher.py`); the HTTP routes are under `/api/queue`.

## Watch a trained agent

```bash
# Latest checkpoint of the run, greedy actions
uv run python scripts/watch.py runs/snake/8x8

# With the attention heatmap, a specific layer, sampled actions (PPO runs; DQN runs always play greedily), faster
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
| `transformer`, `hybrid`, `jepa_attention`, `jepa_tokens` | heatmap of what the class token attends to on the board (`--layer` applies, `--frame` must stay -1/0) | one weight per input number in the window title, e.g. `x 0.12, y 1.00, ...` |
| `temporal_transformer`, `jepa_temporal` | heatmap for one of the history frames (`--layer`, `--frame`) | one weight per number of one history frame (`--frame`) |
| `frame_attention` | one weight per history frame, printed in the window title (`--layer`) | the same |
| `cnn`, `resnet`, `lstm`, `jepa`, `mlp` | nothing: there is no attention, so `--attention` is refused | the same |

## Evaluate a trained agent

An evaluation plays complete games with the best action at every step (PPO: the most likely; DQN: the highest
Q-value; no exploration, no sampling) and
prints the game's statistics: for Snake win rate, steps to win, steps between apples; for Lunar Lander landing,
crash, fly-off and timeout rates and steps to land; for every game scores, episode lengths, returns and how the games
ended. Values that cannot be computed (steps to win without a win) are `N/A`.

```bash
# The latest checkpoint with the run's [evaluation] settings (games, seed, step cap)
uv run python scripts/evaluate.py runs/snake/8x8

# A specific checkpoint, other games, and append the result to the run's evaluations.jsonl
uv run python scripts/evaluate.py runs/snake/8x8 --checkpoint checkpoint_0000500000.pt --games 500 --seed 7 --save

# Lunar Lander: the same command, the run says which game it is
uv run python scripts/evaluate.py runs/lunar_lander/mlp --games 20
```

- Defaults: the run's latest checkpoint and its `[evaluation]` settings (a run whose `config.toml` has no
  `[evaluation]` section uses `games = 100`, `seed = 1000`, `max_episode_steps = 10000`). `--max-steps N`
  changes the per-game step cap. It runs on the run's `train.device`.
- **Reproducing:** game `k` always gets the same seed, and play is greedy, so the same checkpoint, `--games`,
  `--seed` and step cap give exactly the same numbers on the same device. The script compares the result with the
  latest recorded evaluation of that checkpoint with those settings and prints `Identical to ...` or `DIFFERENT
  from ...` (or that there is none to compare with).
- `--save` appends the result to `runs/<game>/<name>/evaluations.jsonl` with `"source": "manual"`.

Evaluation while training is set in the `[evaluation]` section of an experiment config:

| Key | Meaning |
| --- | --- |
| `interval` | Evaluate every this many env steps and once at the end of training; 0 = off (the default for configs without the section; `configs/<game>/default.toml` turns it on). Each evaluation first saves a checkpoint at that step. |
| `games` | Games per evaluation. |
| `seed` | Seed of the evaluation games, independent of `train.seed`. |
| `max_episode_steps` | A game still going after this many steps is stopped with the outcome `step_limit` (for games without a time limit of their own). |

Evaluating does not influence training (fresh environments, no sampling, the random streams are restored): a run
with evaluation ends with exactly the same weights as the same run without. With `train.keep_checkpoints = false`
an evaluation's checkpoint is deleted by the next save, so it can no longer be reproduced; the app marks those.

## Where configs live

| What | Where |
| --- | --- |
| Default experiment of a game (every setting except the rewards, commented; PPO) | `configs/<game>/default.toml` |
| The game's defaults for DQN, laid over default.toml (its `[dqn]` section and the shared values that differ) | `configs/<game>/algorithms/dqn.toml` |
| Your own experiments | copy it to e.g. `configs/mine.toml`, pass with `--config` |
| Named reward schemes of a game; `default` holds its default reward values | `configs/<game>/rewards/<name>.toml` (flat file of the `[reward]` keys, no header) |
| The exact config of one run | `runs/<game>/<name>/config.toml` |

An experiment file has the sections `[game] [reward] [model] [train] [<algorithm>] [env] [evaluation]`, where
`[<algorithm>]` is `[ppo]` or `[dqn]`, the one `train.algorithm` names (`[train]` holds what both algorithms
share: budget, gamma, learning rate, seed, device, intervals). `[game] name` says which game it is
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
    metrics.jsonl               one JSON line per logged record (env_steps, scores, the algorithm's losses, ...)
    evaluations.jsonl           one JSON line per evaluation (while training, or evaluate.py --save / the app)
    run.json                    history: one entry per start / resume / fork (see below)
    checkpoint_<steps>.pt       policy + optimizer + env-step counter (+ DQN's target network; never the replay buffer)
```

Metrics are logged each time the env-step count passes a multiple of `train.log_interval` and at the end;
checkpoints likewise for `train.checkpoint_interval` (both count env steps, like `evaluation.interval`). Lower
both for short experiments. Every record has `env_steps`, `iteration`, the episode means (`mean_score`,
`mean_length`, `mean_return`, `episodes`), `learning_rate` and `steps_per_second`, plus the algorithm's own:
PPO `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`, `explained_variance`,
`epochs_completed` (of the latest update); DQN `loss` (Huber, or squared error with `dqn.loss = "mse"`), `q_mean`, `td_error` (means over the gradient
steps since the previous record), `epsilon`, `replay_size`, `gradient_steps` (with `dqn.prioritized` also `priority_beta`, `priority_mean`,
`priority_max`). An iteration is one PPO update
(`rollout_length * num_envs` env steps) or one DQN step of every game (`num_envs` env steps).

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

One line of `evaluations.jsonl` (`statistics` holds the game's own keys first, then the generic ones; `null` = N/A):

```json
{"timestamp": "2026-11-14T10:00:00+00:00", "source": "training", "env_steps": 1000000,
 "checkpoint": "checkpoint_0001000000.pt",
 "settings": {"games": 100, "seed": 1000, "max_episode_steps": 10000, "parallel_games": 32, "device": "cuda", "policy": "greedy"},
 "statistics": {"win_rate": 0.0, "mean_steps_to_win": null, "median_steps_to_win": null,
                "mean_steps_between_apples": 10.96, "median_steps_between_apples": 7.0,
                "games": 100.0, "mean_score": 26.07, "median_score": 28.0, "min_score": 2.0, "max_score": 37.0,
                "mean_length": 290.34, "mean_return": 22.18},
 "outcomes": {"hit_self": 89, "hit_wall": 11}, "seconds": 1.7}
```

`source` is `training` or `manual`; `checkpoint` names the evaluated weights (an evaluation always has one
when the run saves checkpoints).

## Choosing a model kind and history

Set `kind` in `[model]` and `history_length` / `history_actions` in `[env]`.

| `model.kind` | idea on a board (Snake) | on numbers (Lunar Lander) | suggested `[env]` |
| --- | --- | --- | --- |
| `transformer` | each board cell is a token | each number is a token | `history_length = 1` |
| `cnn` | convolutional baseline, no attention | an MLP (nothing to convolve); use `mlp` instead | `history_length = 1` |
| `resnet` | residual conv blocks (two convs and a skip each; `conv_layers` blocks), no attention | residual MLP blocks with layer norm (as in SimBa), `conv_layers` blocks | `history_length = 1` |
| `hybrid` | conv stem, then transformer over cells | per-number MLP stem, then transformer over numbers | `history_length = 1` |
| `temporal_transformer` | attention over the cells of several frames (plus action tokens) | attention over every number of several frames (plus action tokens) | `history_length = 4`, `history_actions = true` |
| `frame_attention` | CNN per frame, attention across frames | MLP per frame, attention across frames | `history_length = 4`, `history_actions = true` |
| `lstm` | CNN per frame, an LSTM over the frames | MLP per frame, an LSTM over the frames | `history_length = 4` or more, `history_actions = true` |
| `jepa` | CNN encoder, a predictor of each action's next-state embedding, a value decoder; acts on the predicted values | the same with an MLP encoder | `history_length = 1` |
| `jepa_attention` | `jepa` with `hybrid`'s encoder (conv stem, transformer over cells, class token = the embedding) | the same with hybrid's per-number stem | `history_length = 1` |
| `jepa_tokens` | `jepa_attention` whose predictor predicts every cell's token of the next state (a small transformer, `predictor_layers` blocks) | the same, one token per number | `history_length = 1` |
| `jepa_temporal` | `jepa` with `temporal_transformer`'s encoder (attention over the cells of several frames plus action tokens, class token = the embedding) | the same over every number of several frames | `history_length = 4` (8 on Lunar Lander with hidden velocities), `history_actions = true` |
| `mlp` | flattened board, weak baseline | plain MLP; recommended | `history_length = 1` |

The non-temporal kinds see stacked frames only as extra input channels (extra values per number). Which `[model]` fields each kind uses
(for example `cnn` ignores `num_heads`) is listed by the app's config page. Every kind works with both
algorithms: the trunk is the same, only the head differs (PPO: a policy head and a value head; DQN: one Q-value
per action), and attention is shown the same way.

`lstm` is a DRQN-style policy with a fixed unroll length. The `history_length` frames (and the past actions, with
`history_actions = true`) are encoded one by one and an LSTM reads them oldest to newest; its output after the newest
frame feeds the head. Its memory is exactly that window: the hidden state starts at zero at every step, is not carried
from one env step to the next and is not stored in the DQN replay buffer. `kind = "resnet"` and `"lstm"` have no
attention, so Watch shows no attention panel for them. `resnet` uses `d_model`, `conv_channels`, `conv_layers` (the number of
residual blocks) and, on boards, `kernel_size`; `lstm` uses those plus `num_layers` (LSTM layers) with `conv_*` building its per-frame encoder.

`model.residual = true` (`hybrid`, `jepa_attention`, `jepa_tokens`, `frame_attention`, `lstm`; off by default, locked when forking) builds their convolutional
stem (an MLP on numbers; for the other two the per-frame encoder) from `resnet`'s residual blocks instead of plain layers, e.g.
`--set model.kind=hybrid --set model.residual=true`. Other kinds ignore it.

`model.layer_norm = true` (`cnn`, `resnet`, `mlp`, `jepa`, `hybrid`, `jepa_attention`, `jepa_tokens`, `frame_attention`, `lstm`;
off by default, locked when forking) adds a normalisation layer after every convolution or fully connected layer of the
part that reads the observation (the conv / MLP trunk, the stem in front of a transformer, the per-frame encoder) and in
its readout, each before the activation. On numbers it is a layer norm; on boards it is a group norm with one group,
which normalises each board's whole feature map at once (the layer-norm equivalent for convolutions; no batch statistics,
so playing, training batches and DQN's target network compute the same function). With it off the network and its
checkpoints are exactly those from before the switch. What it does not touch: the transformer blocks (they have layer
norms already, so `transformer`, `temporal_transformer` and `jepa_temporal` ignore it), the jepa predictor, the LSTM
itself and the heads. Residual blocks on numbers (`resnet` on a vector game, `residual = true`) have layer norms of
their own, so there only the readout gains one, and a `hybrid`-style stem with `residual = true` on a vector game does not change.
E.g. `--set model.kind=cnn --set model.layer_norm=true`.

Memory only matters when the observation hides something. Lunar Lander's `game.hide_velocities = true` shows the model zeros
instead of its three velocity numbers (the physics, score and reward still use the true state), so it has to infer them from
how position and angle change over several frames:

```bash
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/blind-lstm --set game.hide_velocities=true \
    --set model.kind=lstm --set env.history_length=8
```

`hide_velocities` is locked when forking a run (it changes what the observation means).

### jepa: acting on predicted next states

`kind = "jepa"` is a small world model trained from scratch on the game (no pretrained weights). An encoder (the `cnn`
encoder: `conv_channels`, `conv_layers`, `d_model`) turns the observation into an embedding; a predictor (`num_layers`
deep) takes the embedding and each action and predicts the embedding of the next state; a value decoder turns an
embedding into a value. With DQN, Q(s, a) = value of the embedding action a is predicted to lead to, so the agent plays
the action whose predicted next state is worth most. With PPO the policy scores the predicted embeddings (its own
decoder) and the value decoder values the current embedding.

`kind = "jepa_attention"` is the same world model with `hybrid`'s encoder: a conv stem (an MLP per number on vectors),
a transformer over the cells (numbers) and the class token's output as the embedding. Its `num_layers` sets both the
number of transformer blocks and the predictor's depth; `num_heads`, `mlp_ratio`, `dropout` and `residual` work as for
`hybrid`, and `watch.py --attention` / the Watch tab show its encoder's attention.

`kind = "jepa_tokens"` keeps every token of that encoder instead of only the class token: the state embedding is the class
token plus one token per cell (per number). Its predictor is a small transformer (`model.predictor_layers` blocks, default 1;
`num_layers` is the encoder's) that adds the action's embedding to every token and predicts the whole token set of the next
state, cell by cell; the decoders read the class token of a (predicted or current) token set, so the heads below work as for
the other two. The prediction loss compares the token sets token by token. The predictor runs once per action, so it is
slower than `jepa_attention`. `predictor_layers` is read by no other kind and is locked when forking.

`kind = "jepa_temporal"` is the same world model with `temporal_transformer`'s encoder: a token for every cell (number) of each
of the last `history_length` frames plus one per past action (`history_actions = true`), transformer blocks, and the class
token's output as the embedding, so the embedding can hold what one frame does not show (Lunar Lander's velocities with
`game.hide_velocities = true`). Like `jepa_attention`, its `num_layers` sets both the transformer blocks and the predictor's
depth; it has no conv stem (no `conv_*`, no `residual`), and `watch.py --attention` / the Watch tab show its encoder's
attention per frame as for `temporal_transformer`. Everything below applies to all four kinds:

```bash
uv run python scripts/train.py --run-dir runs/snake/jepa-attention --set train.algorithm=dqn --set model.kind=jepa_attention
uv run python scripts/train.py --run-dir runs/snake/jepa-tokens --set train.algorithm=dqn --set model.kind=jepa_tokens
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/blind-jepa-temporal --set game.hide_velocities=true \
    --set model.kind=jepa_temporal --set env.history_length=8 --set env.history_actions=true
```

Six `[model]` settings belong to them, and unlike the rest of `[model]` they may change when forking a run:

- `prediction_coef` (default 0.0 = off): weight of the prediction loss, which makes the embedding predicted for the action
  taken match a slowly updated (EMA) copy of the encoder's embedding of the real next state. Above 0 it is on; measurements
  so far showed it does not beat leaving it off and costs 20-35% of the speed. The other settings below only matter then. The run then logs no `prediction_loss` / `embedding_std` / `value_gap`.
- `prediction_trains_encoder` (default false): whether that loss also trains the encoder. Off, it trains the predictor
  only and the encoder learns from the reinforcement-learning loss alone (it matters only with `prediction_coef > 0`). On was the old default; in a single-seed DQN
  test on Lunar Lander it made the agent end up flying off (README, Caveats).
- `target_ema` (default 0.99): how slowly that target encoder follows the trained one (per gradient step).
- `rollout_steps` (default 1): how many steps in a row the prediction loss follows. 1 is the plain loss (predict the
  next state from the real current one). With K above 1 the predictor is also fed its own prediction, along the K
  actions that were really taken, and every one of those predictions is compared with the state the game really
  reached; a sequence stops where its episode ended (and, with PPO, at the end of the rollout; with DQN at the newest
  step in the replay buffer). This is practice for looking several steps ahead. It costs speed: 3 steps trained about a
  fifth slower than 1 (jepa_tokens, Snake 6x6, DQN, one run).
- `rollout_value_coef` (default 1.0; DQN without `dqn.dueling`, and only with `rollout_steps > 1`): weight of a second
  part of that loss. From every predicted state the model reads the value of each action exactly as a lookahead would
  (the predictor applied to its own prediction, then the value decoder), and this part pulls those values to the ones
  the model gives the real state. It trains only the predictor's step from a predicted state; the encoder, the value
  decoder and the predicted embedding itself get no gradient from it. 0 switches it off. PPO and the dueling head
  ignore it (their value decoder reads the current state, so there are no such action values).
- `value_training` (DQN only, default `predicted`): `predicted` trains the decoded predicted embedding of the action
  taken with the TD loss; `actual` trains the decoded embedding of the next state that really happened (the value of
  arriving there, its reward included) and leaves the predictor to the prediction loss, which then matches whole
  embeddings (mean squared distance instead of cosine distance) and also the decoded values of prediction and real
  next state. `actual` needs `prediction_coef > 0` (the default 0 is rejected) and did not learn in our tests: the predictor's error in the value
  is as large as the value differences between actions (README explains it).

```bash
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/jepa --set train.algorithm=dqn --set model.kind=jepa
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/jepa-loss --set train.algorithm=dqn \
    --set model.kind=jepa --set model.prediction_coef=1
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/jepa-actual --set train.algorithm=dqn \
    --set model.kind=jepa --set model.prediction_coef=1 --set model.value_training=actual
uv run python scripts/train.py --game lunar_lander --run-dir runs/lunar_lander/jepa-per --set train.algorithm=dqn \
    --set model.kind=jepa --set dqn.prioritized=true
uv run python scripts/train.py --run-dir runs/snake/jepa-rollout --set train.algorithm=dqn \
    --set model.kind=jepa_tokens --set model.prediction_coef=1 --set model.rollout_steps=3
```

The target encoder is saved with the checkpoint's training state (not with the policy weights), so resuming and forking
keep it; watching and evaluating do not need it. The Runs chart offers `prediction loss`, `embedding spread` (near 0
would mean every state gets the same embedding) and `value gap` (how far the value of the predicted next state is from
that of the real one; it matters for `actual`, which acts on it) for jepa runs with `prediction_coef > 0`, and with `rollout_steps > 1` also `rollout last-step loss` (the
prediction loss at the last step of the rollout alone) and, under DQN, `chained value gap` (how far the action values
read from a predicted state are from those of the real state: the error a lookahead through the predictor makes). Prioritized
replay (`dqn.prioritized = true`) works with every `value_training`: the priorities are the TD errors of the values the
TD loss trains (Q(s, a) for `predicted`, the real next state's value for `actual`); the prediction loss is not
importance-weighted and does not change the priorities.

## Choosing an algorithm

| `train.algorithm` | what it is | its settings that matter most |
| --- | --- | --- |
| `ppo` | on-policy actor-critic: learns from rollouts it has just played, a few epochs each | `rollout_length`, `num_minibatches`, `entropy_coef`, `gae_lambda` |
| `dqn` | off-policy, value-based (Double DQN): plays epsilon-greedily into a replay buffer, learns from random batches against a target network | `learning_rate`, `train_frequency`, `target_update_interval`, `epsilon_fraction`, `buffer_size` |

Prioritized replay (`dqn.prioritized = true`, Schaul et al. 2016, proportional variant): a transition is drawn with
probability proportional to `(|TD error| + priority_epsilon) ** priority_alpha`, new transitions get the largest
priority so far, and each sample's loss is scaled by an importance weight `(N x probability) ** -beta` (normalised
by the batch maximum) with `beta` rising linearly from `priority_beta_start` to 1 over `train.total_env_steps`.
It adds a little bookkeeping per step and can be forked with different priority settings.

n-step targets (`dqn.n_step`, default 1): the target of a transition is the discounted sum of the next `n_step`
rewards plus `gamma ** n_step` times the bootstrap value of the state reached after them. The window stops at an
episode's end (a time-limit cut is bootstrapped from its own final state when `train.bootstrap_truncated` is true,
otherwise it counts as an end) and at the newest step in the buffer. The buffer still stores single steps and joins
them when a batch is sampled, so it needs no extra memory, works with prioritized replay (the priority is the TD
error of the n-step target) and can be changed on fork. No correction is made for the later steps having been
chosen by an older, exploring policy.

Dueling head (`dqn.dueling`, default false): the head has a state-value stream and an advantage stream (one hidden
layer of `d_model` units each) and outputs `Q = V + A - mean(A)`. Every model kind supports it; the jepa kinds read
`V` from the current embedding and the advantages from their predicted embeddings, and refuse it together with
`model.value_training = "actual"`. It changes the network, so it is fixed for a run: `--fork` with a different
`dqn.dueling` is refused, `--resume` reads it from the run's `config.toml`.

Loss (`dqn.loss`, `"huber"` by default or `"mse"`): how the error between Q(s, a) and its target is penalised.
Huber is squared below an error of 1 and linear above; `mse` is squared throughout (twice Huber's gradient for
small errors, unbounded for large ones). The optimizer is Adam either way.

DQN's replay buffer lives in memory: about `2 x buffer_size x (numbers in one observation) x 4` bytes (Snake 8x8,
100k transitions: ~150 MB; Lunar Lander: ~6 MB; history frames multiply it). It is not saved.

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

Serves http://127.0.0.1:8000: configure (including the algorithm), start, queue and resume training, continue a run under a new name with changed
settings, evaluate checkpoints and chart evaluations, watch an agent with its attention, view the architecture and
edit reward schemes.
