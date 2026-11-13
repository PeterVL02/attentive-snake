# Attentive Snake

RL playground: train policies (mainly transformers) to play Snake with PPO and
inspect them (attention heatmaps, architecture diagrams, run comparison). The
main entry point for the owner is the localhost app; the CLI does the same things.

## Commands

```bash
uv run pytest -q                              # full suite, ~15 s, must stay green
uv run python -m attentive_snake.app          # app on http://127.0.0.1:8000
uv run python scripts/train.py --run-dir runs/NAME [--config FILE]
uv run python scripts/train.py --resume runs/NAME [--steps N]
uv run python scripts/watch.py runs/NAME [--attention]
uv run python -m attentive_snake.game.play    # human play (pygame)
```

Python 3.14, uv, src layout. `pygame` is `pygame-ce`. Headless pygame checks:
`SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 timeout 6 ...` (exit 124 is success).
`docs/CLI.md` is the user-facing cheat sheet and is displayed in the app; keep
it in sync with the script flags.

## Design rule (the owner's, and it is the main review criterion)

Every class, function and module has as few responsibilities as possible.
Prefer a new small module over growing an existing one. Packages talk through
small contract files, not through each other's internals:

- `game/types.py` — `Action`, `GameState`, `StepEvent`, `StepResult`
- `env/types.py` — `VecEnv` protocol, `VecEnvStep`, `EnvStep`, `EpisodeSummary`
- `models/types.py` — `PolicyOutput(logits, value)`

## Package map (`src/attentive_snake/`)

- `game/` — `engine.py` is pure rules (no pygame/numpy/torch); `renderer.py`
  only draws a `GameState` (+ optional heatmap); `play.py` is human play.
- `env/` — numpy only, no torch, no pygame.
  - `observation.py`: `GameState` → `(3, H, W)` head / body / food. The body
    channel holds time-until-vacated, which encodes order and heading.
  - `frame.py`, `history.py`: optional stacking of the last K frames (and 4
    one-hot action planes per frame). Observation is `(K*F, H, W)`, oldest
    frame first.
  - `reward_terms.py`, `reward_pipeline.py` (`RewardBuilder`), `reward.py`
    (`RewardConfig`, `build_reward`): rewards are a sum of small terms; a term
    whose coefficients are zero is not added.
  - `snake_env.py` single env; `vec_env.py` `SyncVecSnakeEnv` (Python loop,
    auto-reset, per-env seeds).
- `models/` — five kinds in `registry.py` (`build_policy`, `KIND_INFO`):
  `transformer`, `cnn`, `hybrid`, `temporal_transformer`, `frame_attention`.
  - Every policy: `(config, observation_shape, num_actions, history_length)`,
    whole architecture under `self.net` (the diagram tool reads it), forward
    returns `PolicyOutput`. Input shape comes from the env, never from
    `ModelConfig`.
  - Attention kinds subclass `AttentivePolicy` (`policy_base.py`) and expose
    `attention_maps(obs)` (per block `(B, heads, T, T)`) and `token_layout`.
    Viz and watch code dispatch on those, never on class names.
  - Adding a kind: implement the policy, register it and its `KindInfo`
    (which `[model]` fields it uses — `tests/test_schema.py` checks this is
    truthful).
- `training/` — PPO. `rollout.py` (buffer, `compute_gae`), `collector.py`,
  `ppo.py` (pure loss + update), `metrics.py` (loggers, `metrics.jsonl`),
  `checkpoint.py`, `resume.py`, `schedule.py`, `runtime.py`; `trainer.py` only
  orchestrates. Library code depends on the contracts only, not on concrete
  envs or models.
- `config/` — `ExperimentConfig` = `[game] [reward] [model] [train] [env]`,
  TOML in/out (`experiment_io.py`, own tiny writer; `None` is omitted),
  `schema.py` (JSON schema for the app), `reward_schemes.py`, `reward_table.py`.
  Field descriptions live in each dataclass field's `metadata["help"]` — the
  single source of truth; every new field needs one and a line in
  `configs/default.toml` (tests enforce both).
- `viz/` — `tree.py` → `text.py` / `mermaid.py` / `html.py`; `attention.py`
  turns attention maps into heatmaps.
- `watch/` — `WatchSession`: headless playback of a checkpoint producing
  JSON-able frames with heatmaps; used by both `scripts/watch.py` and the app.
- `factory.py` — builds env + policy from an `ExperimentConfig`.
- `run_dir.py` — layout and discovery of `runs/<name>/`.
- `app/` — FastAPI backend + static frontend (plain ES modules, no build step,
  no CSS framework). Training runs as a subprocess of `scripts/train.py` via
  `jobs.py`; `routes/` has one module per concern; `static/js/tabs/` one module
  per tab.

## Things to know

- **Done/GAE indexing:** `dones[t]` belongs to the step that produced it; envs
  auto-reset, so `observations[t+1]` is already the next episode. Truncation
  (starvation) is treated as termination.
- **Run directory is the source of truth for configs.** Checkpoints hold only
  weights, optimizer and step count. `--resume` reads `runs/<name>/config.toml`.
- **Resume** re-stretches the LR decay over the new total and does not restore
  env state or RNG streams.
- **Code defaults vs `configs/default.toml`:** the file is 8x8 with a
  starvation limit; `GameConfig()` is 10x10 with none.
- **Dropout must stay 0** for PPO; collection runs in eval mode.
- **The app's UI cannot be verified from a terminal.** Backend changes are
  covered by `tests/test_app.py`; say plainly when a frontend change was not
  seen in a browser and ask the owner to check it.
- **Do not delete or modify the owner's runs** under `runs/` (git-ignored).
  Use throwaway names for test runs and remove them afterwards.
- Dependencies are added with `uv add`; nothing else edits `pyproject.toml`.

## Working style the owner asked for

Be usage-aware: the lead delegates mechanical work and code-writing to
subagents (sonnet or opus), gives them exact interfaces to code against, and
has them report what they built, why and how it works. The lead does not need
to review every line but verifies the suite and an end-to-end run.
