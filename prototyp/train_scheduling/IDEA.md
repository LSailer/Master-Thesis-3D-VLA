# train_scheduling — when should train_step run relative to env stepping?

## Goal

Compare three schedules for interleaving environment stepping (Habitat) and
agent training (`agent.train_step`) using the REAL stack (real env, real
`R2DreamerAgent`, real `ReplayBuffer`), and measure which one yields the best
wall-clock throughput at a fixed train-ratio:

1. **interleaved** (reference) — the current production schedule from
   `train_loop()` (src/r2dreamer/launch/loops.py:401): every env step accrues
   `train_ratio / (batch_size * seq_len)` credit and train steps run inline
   whenever credit >= 1.
2. **episode** — env-step until the episode ends (`result.done`), then run all
   accumulated train-step credit at the episode boundary (plus a final drain).
   Same credit arithmetic, just deferred.
3. **threaded** — two Python threads: an actor thread (act -> env.step ->
   buffer.add via a `LockedReplayBuffer` wrapper) and a learner thread
   (buffer.sample -> `agent.train_step`), with a stop event. The learner paces
   itself against the actor's credit-eligible step counter so the train-ratio
   is maintained approximately.

## Hypothesis

The interleaved loop serializes two workloads that mostly wait on different
resources: env stepping is dominated by habitat-sim's C++ physics/render (CPU
+ GL), while `train_step` is dominated by JAX GPU compute. Both should release
the CPython GIL for their expensive parts (habitat's C++ step via pybind11,
JAX during device execution/dispatch waits), so two plain Python threads
*should* overlap them and approach

    wall_threaded ~= max(wall_env, wall_train)  <  wall_env + wall_train

**This is exactly what the measurement must decide** — if habitat or JAX holds
the GIL during their hot paths, the threaded mode will degenerate to
interleaved timing (overlap gain ~1.0x) and the experiment falsifies the
hypothesis cheaply, before any production-loop surgery.

Episode-boundary training is not expected to be faster in total (same work,
same serialization); it is the control that isolates *scheduling granularity*
from *parallelism*: interleaved vs episode measures batching/locality effects,
threaded vs interleaved measures true overlap.

## Approach

- **Wrap, don't edit `src/`**: `locked_buffer.LockedReplayBuffer` delegates to
  the real `ReplayBuffer` under a `threading.Lock` (the buffer's `add()` /
  `sample()` are not thread-safe: unguarded `idx`/`size` mutation and ring
  gather). The wrapper is swapped into the real `ExperienceCollector.buffer`
  attribute for threaded mode only.
- **Reuse the composition root**: `run_scheduling_experiment.py` builds env,
  encoder/adapter, agent, and collector by importing the same helpers
  `src.r2dreamer.launch.train.train()` uses (`_make_encoder_bundle`,
  `_make_env_instances`, `_make_agent_config`, `_make_collectors`, ...) and
  selects the run via a canonical `scripts/r2dreamer/_run_configs.py` run id
  (default `habitat-l1-cnn` — lightest adapter, no VGGT weights).
- **Timing loops** live in `scheduling_loops.py` — stripped of logging/val/
  checkpoint cadences so all three modes time exactly act+step+sample+train.
  Prefill (real `loops.prefill`) runs before the timed section in every mode.
- Per-mode results land in `outputs/prototype/train_scheduling/<run>/`
  as `results_<mode>.json` plus an accumulating `MANIFEST.json`
  (status ok/failed — the manifest, not the exit code, judges the run; habitat
  GL teardown poisons exit codes on this cluster). `--summarize` computes the
  threaded-vs-interleaved overlap gain.

## Metrics

Per mode: total wall time (s), env-steps/s, train-steps/s, achieved
train/env ratio. Derived: `overlap_gain = wall_interleaved / wall_threaded`.

## Success criteria

- All three modes complete a smoke run (600 steps, prefill 200, B=4 T=16,
  train_ratio 16) with `status: ok` in the manifest and matching train-step
  counts (interleaved == episode exactly; threaded within one batch of target).
- A clear overlap-gain number: > ~1.3x says a threaded production loop is
  worth graduating; ~1.0x says the GIL (or a shared resource) blocks overlap
  and the idea dies here cheaply.
