# ADR 0006: Decouple environment and replay buffer from the Trainer

Status: proposed (design only — no code changed yet)
Date: 2026-07-22

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

## Open questions

- Naming: `ExperienceCollector` / `ExperienceSource` vs `RolloutWorker` —
  pick before implementation.
- Should `EpisodeSummary.video_frames` be raw composed frames (trainer calls
  `log_episode_video`) or should the collector take a frame-sink callback?
  Proposal: raw frames — keeps W&B out of the collector.
