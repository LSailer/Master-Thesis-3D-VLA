# ADR 0006: Decouple environment and replay buffer from the Trainer

Status: superseded 2026-07-28 by the src.main orchestrator refactor - the
loop functions described here (`src/r2dreamer/launch/loops.py`) moved into
`src/main.py` (`run_loop`/`prefill`/`overfit`), `train_step` and `act` became
directly jitted pure entry points with explicit state, and the snapshot/restore
acting-state dance was replaced by a functional carry. The decoupling this ADR
argued for (env/replay outside the agent) still stands.
Previously: implemented 2026-07-22 (golden-equivalence verified: bit-identical
metrics.csv vs. the pre-refactor Trainer on a fixed-seed 40-step run)
Date: 2026-07-22

Implementation deviations from the design below:

- ``sample()`` returns the adapter-augmented ``ReplayBatch`` struct, not a
  packed array dict — that is what ``agent.train_step`` consumed all along;
  ``replay_batch_to_arrays`` moved to ``src/buffer/replay_arrays.py`` as a
  utility.
- The collector gained ``auto_reset`` (constructor) and ``summarize``
  (per-step) switches plus ``finish_episode()``: validation must not
  auto-reset (an extra reset advances Habitat's pinned episode iterator) and
  prefill must not fire the episode-metrics fn (it mutates rolling SR
  trackers).
- Loops live in ``src/r2dreamer/launch/loops.py`` (imported by the main
  module ``launch/train.py``); ``apply_resume`` is a loops function called
  from the launcher. ``train()`` returns a ``TrainingRun`` handle instead of
  the deleted ``Trainer``.

## Context

`src/r2dreamer/trainer.py` currently owns five distinct responsibilities:

1. **Loop orchestration** — prefill → train-ratio loop → val schedule →
   checkpoint/manifest/W&B/CSV.
2. **Environment interaction** — `env.reset()/step()`, scene-aware
   `on_episode_reset(scene_id)` hooks, episode bookkeeping
   (reward/steps/action-counts), auto-reset on `done`.
3. **Observation preparation** — calling `ObsAdapter.prepare_env_step` to
   split each frame into `replay_obs` (buffer) and `encoder_obs` (acting).
4. **Replay ownership** — constructing `ReplayBuffer`, `add()` per step,
   `sample()` + `augment_replay_batch()` per train step, plus the
   `replay_batch_to_arrays` packing helpers (~150 lines of trainer.py).
5. **Env-coupled diagnostics** — top-down video capture (needs
   `env.agent_state`, `render_topdown_frame(env, …)`), adapter
   diagnostics/growth history, episode metrics callbacks.

Only (1) is genuinely "training". The rest forces every test and every new
env/encoder combination through the full Trainer constructor, and blocks
future work (offline/replay-only training, async collection).

### The encoder boundary (key constraint)

The pipeline has a **frozen** stage and a **trainable** stage:

    env frame ──► ObsAdapter (frozen: VGGT extractor, house-point buffer,
                  symlog/normalize) ──► encoder inputs
    encoder inputs ──► Encoder Module (trainable: params["encoder"] —
                  MLP / PointNet / GNN / hybrid) ──► embedding ──► RSSM …

Gradients must flow into `params["encoder"]`, so the trainable encoders run
*inside* `agent.train_step`. Therefore the batch handed to the trainer must
contain **frozen-stage outputs (encoder inputs)**, not final embeddings.
"The trainer only gets encoded observations" is interpreted as:

- **Acting path**: trainer receives ready-to-encode `encoder_obs` per step
  (already true today via `PreparedObservation`).
- **Training path**: trainer receives array batches whose observation leaves
  are frozen-stage features, already adapter-augmented and `(B, T)`-packed.
- **Encoder params**: remain inside `agent.params["encoder"]`, updated by
  `agent.train_step` and checkpointed via the agent. The trainer never sees
  raw env frames; the collector never sees trainable parameters.

## Decision

Introduce one new abstraction, `ExperienceCollector`, that owns env +
adapter + (optional) buffer behind a small protocol. Loop orchestration
moves out of the package into the main module (`launch/train.py`) as plain
functions; logging plumbing is bundled in a `RunLogger`; the `Trainer`
class is dissolved.

### New module `src/r2dreamer/experience.py`

```python
@dataclass(frozen=True)
class AgentStep:
    """What the agent needs to act: prepared encoder inputs + boundary flag."""
    encoder_obs: Any          # frozen-stage features (dict or array)
    is_first: bool

@dataclass(frozen=True)
class EpisodeSummary:
    """Returned once per finished episode; env internals stay inside."""
    metrics: dict[str, Any]   # episode_metrics_fn already applied
    reward: float
    steps: int
    action_counts: np.ndarray
    video_frames: list[np.ndarray] | None   # only when capture was started

@dataclass(frozen=True)
class StepResult:
    agent_step: AgentStep     # next obs to act on (post auto-reset if done)
    reward: float
    done: bool
    episode: EpisodeSummary | None   # set exactly when done


class ExperienceSource(Protocol):
    """Everything the Trainer may touch. No env, adapter, or buffer types."""
    def reset(self) -> AgentStep: ...
    def step(self, action: int) -> StepResult: ...
    def sample(self, batch_size: int, seq_len: int) -> ReplayArrayBatch: ...
    @property
    def buffer_size(self) -> int: ...            # 0 when recording is off
    @property
    def supports_video(self) -> bool: ...
    def start_video_capture(self) -> None: ...   # frames arrive in EpisodeSummary
    def diagnostics(self) -> dict[str, float]: ...
    @property
    def growth_history(self) -> list[tuple[int, int]]: ...
    def close(self) -> None: ...
```

`ExperienceCollector` is the concrete implementation:

- **Owns**: `Env`, `ObsAdapter`, `ReplayBuffer | None`, `EpisodeMetricsFn`,
  video capture state. The `Env` protocol and `ObservationFrame` move out of
  the trainer's import surface into this module.
- **`step(action)`**: steps the env, runs `prepare_env_step`, records the
  transition into the buffer (when one is attached — the
  `previous_action` consistency check moves here), and on `done`
  auto-resets: fires `on_episode_reset(scene_id)`, builds `EpisodeSummary`
  (metrics fn, video frames), and returns the *new* episode's first
  `AgentStep`. All episode bookkeeping leaves the trainer.
- **`reset()`**: reset + scene hook, never records (matches today: reset
  frames are never added to the buffer, in prefill or train).
- **`sample(B, T)`**: `buffer.sample` → `adapter.augment_replay_batch` →
  `replay_batch_to_arrays`. The packing helpers
  (`replay_batch_to_arrays`, `_stack_*`) move from `trainer.py` to
  `src/buffer/` next to `ReplayBuffer` — they are replay-domain code.
- **Val instance**: `ExperienceCollector(val_env, val_adapter,
  buffer=None, metrics_fn=val_metrics_fn)` — recording is simply absent,
  `sample` raises. This replaces the current three nullable
  `val_*` constructor arguments with one nullable collector.

### Loop orchestration moves to the main module — the Trainer class dissolves

Orchestration is application-level composition, so the loops live in the
main module (`src/r2dreamer/launch/train.py`) as **plain functions**, not
methods on a stateful class. A class that holds agent + collector + configs
just to run a loop re-creates the god object one level up; functions taking
protocol-typed arguments keep every dependency explicit and stubbable:

```python
# launch/train.py (or a sibling loops.py it imports)

def prefill(experience, num_steps, num_actions, rng_key) -> Key: ...
def train_loop(agent, experience, val_experience, acfg, tcfg,
               logger, rng_key, start_step) -> Key: ...
def val_loop(agent, val_experience, tcfg, logger, rng_key, step) -> Key: ...
def overfit_loop(agent, experience, tcfg, logger, rng_key) -> Key: ...
```

`train_loop` keeps exactly what was irreducible in the old
`Trainer._train_loop`: act → `experience.step` → train-ratio credit →
scheduled val/checkpoint. Its body reduces to:

```python
step_out = experience.step(action)                 # record + auto-reset inside
if step_out.episode is not None:
    logger.log_episode(step_out.episode, step)     # csv/wandb/console/video
if experience.buffer_size >= batch_steps:
    while train_credit >= 1.0:
        batch = experience.sample(acfg.batch_size, acfg.seq_len)
        metrics = agent.train_step(batch, key, materialize=will_log)
```

Everything else the old Trainer carried is reassigned:

- **`RunLogger`** (new, small class): CSV writer + W&B init/finish + console
  prints + `MANIFEST.json` start/end + fps bookkeeping (`_t0`,
  `_last_log_*`). One sink object passed into the loops so they never touch
  `wandb` or file handles directly. Adapter end-of-run diagnostics
  (`experience.diagnostics()` / `growth_history`) are flushed here too.
- **Checkpoint/resume**: already free functions in `checkpointing.py`; the
  main module calls `load_checkpoint` during wiring (before the loop starts)
  and `save_checkpoint` on the schedule inside `train_loop`.
- **`run()` scaffold** (try/finally, `hard_exit_on_finish`, status →
  manifest, `close()`): a top-level `run_training(...)` function in the main
  module that composes prefill/train/overfit + logger lifecycle.
- **`snapshot/restore_act_state`** around validation: stays in `val_loop`.

### Wiring (main module)

`launch/train.py` already resolves env + encoder + configs via registries;
it additionally builds the `ReplayBuffer` (taking over `buffer_capacity` /
`float_dtype` / `num_actions` resolution from `Trainer.__init__`), assembles
the train/val collectors, constructs the `RunLogger`, applies resume, and
calls `run_training(...)`. No registry changes; `enc.make_adapter()` /
`enc.new_adapter()` stay as-is. `trainer.py` is deleted once callers
(`launch/train.py`, tests, any notebook entry points) migrate; the
`R2DreamerAgentLike` / `EpisodeMetricsFn` protocols move to
`experience.py` / the loops module.

One discipline to keep: the loops must remain importable functions with
protocol-typed parameters — no logic in an `if __name__ == "__main__":`
block — otherwise moving them "to the main file" would trade the god object
for untestable script code.

## Consequences

- Loop tests no longer need a real env/adapter — a stub `ExperienceSource`
  is a ~30-line fake and the loops are plain functions, so a test calls
  `train_loop(...)` directly with fakes for source and logger. Collector
  tests exercise env/adapter/buffer interplay without W&B/CSV/checkpoint
  scaffolding.
- Offline training = an `ExperienceSource` whose `step` is unused and whose
  `sample` reads from disk. Async collection later = a collector running in
  another process behind the same protocol (actor/learner split), without
  touching the loops.
- `_maybe_log_recon` and hybrid-gate metrics keep working: they consume the
  sampled batch / agent params, both visible inside `train_loop` /
  `RunLogger`.
- Churn: `trainer.py` is deleted (replaced by `experience.py`, `RunLogger`,
  and loop functions in the main module); `launch/train.py`,
  `heldout_eval.py` (if it constructs a Trainer), and trainer tests need
  updates — this is a bigger one-time migration than the slimmed-Trainer
  variant, but ends with no residual Trainer API to maintain. Checkpoint
  format is untouched.

## Alternatives considered

1. **Split `EnvSession` + `ReplayStore`** (rollout vs sampling objects).
   Rejected: `augment_replay_batch` and `prepare_env_step` share one
   *stateful* adapter (live house-context buffer joined into sampled
   batches), so the two objects would need a shared adapter reference —
   hidden coupling for no gain. One collector keeps the state ownership
   obvious.
2. **Trainer holds encoder params separately from the agent** (literal
   reading of "trainer gets the encoder parameters"). Rejected: encoder
   params must be inside the pytree that `agent.train_step` differentiates
   and `optax` updates; splitting them out breaks checkpointing
   (`R2DTrainState`) and the single-optimizer design for no benefit. The
   trainer already "has" them via `agent.params["encoder"]`.
3. **Keep a slimmed `Trainer` class owning the loops** (agent +
   `ExperienceSource` + configs as fields, `run()` as entry). Rejected:
   once env/buffer are behind `ExperienceSource` and logging behind
   `RunLogger`, the class holds no state the loop functions can't take as
   parameters — it would exist only to bundle arguments, and keeps the
   orchestration policy hidden inside `src/r2dreamer/` instead of visible
   at the composition root (the main file), where run-shape decisions
   (prefill → train → val cadence) belong.
4. **Move the whole prefill into the collector**
   (`collector.collect_random(n, rng)`). Deferred: keeping action selection
   in the trainer preserves the invariant that the collector never decides
   actions — one policy-boundary, easier to reason about. Can revisit if
   prefill grows env-specific logic.

## Implementation plan

Incremental, each step lands green on its own so the migration never has a
broken intermediate state:

1. **Move replay packing** — `replay_batch_to_arrays` + `_stack_*` helpers
   from `trainer.py` to `src/buffer/replay_arrays.py`; `trainer.py`
   re-imports them temporarily. Existing tests pass unchanged.
2. **Add `src/r2dreamer/experience.py`** — `AgentStep`, `StepResult`,
   `EpisodeSummary`, `ExperienceSource` protocol, `ExperienceCollector`.
   New unit tests with a scripted fake env + fake adapter covering:
   step→record→buffer growth, `previous_action` mismatch raises, auto-reset
   on done (scene hook fired with `scene_id`, summary returned, new
   episode's first `AgentStep` returned), reset never records, video frames
   only when capture started, `sample` = sample→augment→pack, val-mode
   (`buffer=None`) `sample` raises and `buffer_size == 0`.
3. **Shim step** — rewrite `Trainer`'s internals to construct and delegate
   to an `ExperienceCollector`, public API unchanged. All existing trainer
   tests still pass — this validates the collector inside the real loop
   before the loop itself moves. Run the golden-equivalence check here
   (see Verification).
4. **Extract loops + logger** — add `RunLogger` and
   `launch/loops.py` (`prefill`, `train_loop`, `val_loop`, `overfit_loop`,
   `run_training`). Port trainer tests to loop-function tests against stub
   `ExperienceSource`/`RunLogger`.
5. **Rewire and delete** — `launch/train.py` builds buffer + collectors +
   logger, applies resume, calls `run_training`; delete `Trainer` and the
   shim; migrate `heldout_eval.py` / notebook callers; drop the temporary
   re-imports from step 1.
6. **Cluster validation** — smoke + short prod-shape SLURM run (see below).

## Verification

Three layers, strongest first:

1. **Golden-run equivalence (the refactor is behavior-preserving, so prove
   it):** fixed-seed short run (≈500 steps + prefill, smoke config) on
   `main` vs. the refactor branch; `metrics.csv` must be **identical**
   (modulo the `perf/*` timing rows). This works only if the refactor
   preserves the exact order of env steps, buffer adds/samples, and
   `jax.random.split` calls — treat RNG-threading order as a frozen
   contract during the migration; any reordering breaks comparability and
   must be its own follow-up change. Run at step 3 (shim) and again at
   step 5 (loops extracted). Sort both CSVs before diffing — metrics.csv
   rows are not step-sorted.
2. **Unit/integration tests (CPU, local):** the new collector tests plus
   the ported loop tests; full `pytest` with `JAX_PLATFORMS=cpu` after
   every step. Coverage must include the regression-prone edges: resume
   (skip-prefill path, CSV append mode), overfit-loop pass/fail gate,
   val act-state snapshot/restore, episode boundary at the very first
   step, `done` on the last step of the run.
3. **Cluster smoke (end-to-end):** launch via `scripts/slurm/launch.sh`
   with the standard smoke YAML, judge by `MANIFEST.json` status
   (`completed`) — not the exit code (habitat GL teardown poisons it).
   Then one short production-shape run and compare
   `perf/ms_per_step_interval` against the ~219 ms/step baseline —
   the collector adds one dataclass construction per step, which must stay
   in the noise. Check W&B: episode video logged, val/* metrics present,
   house-buffer growth summary emitted.

## Open questions (resolved at implementation)

- Naming: `ExperienceCollector` / `ExperienceSource` (kept).
- `EpisodeSummary.video_frames` carries raw composed frames; the loop calls
  `logger.log_video` — W&B stays out of the collector.
