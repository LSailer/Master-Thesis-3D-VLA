# RESEARCH: Diagnosing & eliminating JAX recompilation and per-step dispatch overhead in the prefill loop

Scope: `prototyp/house_context_multi_episode/run_multi_episode.py::prefill`, the
`JAXVGGTFeatureExtractor.extract` KV-cache path (`src/vggt/jax/feature_extractor.py`),
and the `HouseContextAdapter` (`src/adapters/house_context.py`). Primary sources:
JAX docs (docs.jax.dev), JAX source (github.com/jax-ml/jax), danijar/dreamerv3 +
embodied, and this repo's code.

---

## TL;DR (ranked, actionable for THIS loop)

1. **The per-step `int(jax.random.randint(action_key, ...))` is the clearest
   JAX-side waste, but it does NOT cause recompiles** — it causes a per-step
   device dispatch **plus a host<-device sync** (the `int(...)`). Fix: pre-sample
   all `prefill_steps` actions once with **NumPy** on the host, outside the loop
   (this is exactly what DreamerV3's `RandomAgent` does — it samples with
   `np.stack([v.sample() ...])`, not `jax.random`). Removes N device round-trips.
   `run_multi_episode.py:43-44`.
2. **Confirm the VGGT budgets are actually static.** With the default
   `JAXVGGTFeatureExtractor()` constructor the code fixes budgets to
   `(_max_budget,) * _agg_depth` (`feature_extractor.py:377-378`), so the
   `static_argnums` budget tuple does NOT change per frame and the aggregator
   compiles **twice** (frame-0 vs frame-N, gated by the `is_first_frame` bool at
   `feature_extractor.py:733`). If anyone passes `budgets_static=None`-equivalent
   dynamic budgets via `_compute_static_budgets`, every frame with new scores
   recompiles the aggregator — verify this is off before profiling.
3. **Turn on recompile detection first, before optimizing anything**:
   `JAX_LOG_COMPILES=1` + `JAX_EXPLAIN_CACHE_MISSES=1`. If prefill prints only a
   handful of compile lines total (not ~one per step), recompilation is NOT your
   problem and you should stop here and profile wall-clock sections instead.
4. **The likely real bottleneck is Habitat `env.step` + sensor render and the
   VGGT forward itself, not recompilation.** Memory note `house-points-pose-step-cost`
   already attributes ~219 ms/step to VGGT (132 ms) + replay sample (59 ms).
   Add per-section `time.perf_counter()` timers (env.step vs extract vs adapter
   vs buffer.add) and/or one `jax.profiler.trace(..., create_perfetto_trace=True)`
   window to attribute the cost.
5. **Enable the persistent compilation cache** so warmup compiles survive across
   process restarts. The extractor already sets
   `JAX_COMPILATION_CACHE_DIR=/tmp/vggt_jax_cache` (`feature_extractor.py:26`), but
   lower `jax_persistent_cache_min_compile_time_secs` to `0` so short graphs are
   also cached, and set the dir to shared/persistent storage (not `/tmp`).
6. **Remove remaining per-frame host syncs inside `extract`**: the camera-head
   capacity check does `int(jnp.asarray(valid_len))` per block per frame
   (`feature_extractor.py:791`) and `_run_heads` does
   `int(jnp.asarray(patch_start_idx))` per frame (`feature_extractor.py:826`).
   Each is a blocking device->host transfer. `patch_start_idx` is always 5 —
   hoist it to a constant. Only relevant when `compute_heads=True`.

---

## 1. Detecting recompiles (exact APIs, verified)

| Tool | How to enable | What it tells you | Source |
|---|---|---|---|
| `jax_log_compiles` | `jax.config.update("jax_log_compiles", True)` or `JAX_LOG_COMPILES=1` | Logs elapsed tracing/lowering/compilation time as a **warning** for every compilation. One line per compile => count compiles. | [Debugging slow tracing/compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) |
| `jax_explain_cache_misses` | `jax.config.update("jax_explain_cache_misses", True)` or `JAX_EXPLAIN_CACHE_MISSES=1` | Logs an **explanation of WHY** JAX missed the in-memory tracing cache (and, eventually, the persistent cache), including the Python line and which argument changed (shape/dtype/weak_type/static value). Currently implemented for tracing-cache misses. | [Config options](https://docs.jax.dev/en/latest/config_options.html); [Debugging slow tracing/compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) |
| `jax.log_compiles()` context mgr | `with jax.log_compiles(): ...` | Same as the flag, scoped to a block. | [Debugging slow tracing/compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) |
| jit cache size | `jitted_fn._cache_size()` (internal attr on the `JitWrapped`) | Integer count of compiled variants currently cached for that function. A number that keeps rising across steps == recompiling. NOTE: this is an **internal/underscore** method, not in the public API docs. | JAX source `jax/_src/pjit.py` (jit wrapper); not documented on [jax.jit page](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) |
| `jax.make_jaxpr(f)(*args)` | call it | Prints the traced jaxpr for given arg shapes; use to confirm two call sites trace to identical shapes/dtypes. | [Config options](https://docs.jax.dev/en/latest/config_options.html) |
| `JAX_DUMP_IR_TO` + `JAX_DUMP_IR_MODES=eqn_count_pprof` | env vars | Dumps IR + a pprof-compatible equation-count profile (flame graph of where tracing time goes). | [Debugging slow tracing/compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) |

Recommended one-shot diagnostic for this loop: run `prefill` with
`JAX_LOG_COMPILES=1 JAX_EXPLAIN_CACHE_MISSES=1` for ~60 steps and count compile
lines. Expected healthy result with the extractor's fixed budgets: a burst of
compiles during `_warmup`/frame-0/frame-1, then **silence**. Any compile line
appearing steadily per step is a bug — read the `explain_cache_misses` reason.

## 2. Root causes of recompiles (and which apply here)

JAX caches a compiled executable keyed by (function identity, input avals =
shape+dtype+weak_type, static_argnums *values*, donation, device/backend). A miss
on any component retraces+recompiles. ([Debugging slow tracing/compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html), [Config options](https://docs.jax.dev/en/latest/config_options.html))

- **Changing array shapes/dtypes** — the classic cause. In this repo the KV cache
  is stored **pre-padded to fixed `_cache_max`** with a separate integer
  `valid_len` (`feature_extractor.py:159, 436-448, 580-581`, docstring
  `feature_extractor.py:282` "3-tuples of (k, v, valid_len) for JIT stability").
  So the padded k/v shapes are constant across frames -> **no shape-driven
  recompile from cache growth**, which is the correct design. The growing content
  is tracked by the scalar `valid_len`, not by a growing dimension.
- **`static_argnums` value changes** — the aggregator jit uses
  `static_argnums=(3,4,6,7)` (`feature_extractor.py:224`): arg3 `is_first_frame`
  (bool -> at most 2 variants), arg4 `total_budget` (constant), arg6 `use_cache`
  (constant True), arg7 `current_budgets_static` (the per-block budget tuple).
  **If `budgets_static` is fixed (default), this is a constant tuple => no
  recompile.** If dynamic budgets are enabled, arg7 changes whenever
  `_calculate_dynamic_budgets(last_scores)` returns a different tuple ->
  **recompile per distinct budget vector**. This is the single biggest latent
  recompile risk in the extractor. Default path is safe
  (`feature_extractor.py:377-378, 716-720`).
- **weak_type promotion from Python scalars** — passing a raw Python `int`/`float`
  into a jitted fn gives a `weak_type=True` aval; passing a concretely-typed
  array gives `weak_type=False`. Mixing the two across calls retraces. Not a
  current problem here (the extractor feeds typed arrays), but the pre-sampled
  action array (fix #1) should be a fixed-dtype `np.int32` array to be safe if it
  ever feeds a jitted step.
- **New callable identities per call** — lambdas / `functools.partial` /
  closures created inside the loop are new objects each iteration and defeat the
  cache. Fix: define jitted fns at module/`__init__` scope. The extractor already
  does this (jit built once in `__init__` via the `_make_*` factories,
  `feature_extractor.py:197, 224, 242`). Do NOT introduce per-step `jax.jit(...)`
  or `partial` in the prefill loop. ([Debugging slow tracing/compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html), "avoid recreating function objects; use functools.partial not lambda").
- **Donated buffers** — donation changes the cache key; inconsistent donation
  retraces. Not used in this loop.
- **Per-episode cache reset changing shapes** — `reset()` reallocates the padded
  cache to the **same** `_cache_max` shape (`feature_extractor.py:436-448, 587`),
  so an episode boundary does not change avals -> no recompile. Good.

## 3. Fixes (verified APIs)

### 3a. Pre-sample random actions on the host (biggest single win here)
`run_multi_episode.py:43-44` currently does, per step:
```python
rng_key, action_key = jax.random.split(rng_key)      # device op + key management
action = int(jax.random.randint(action_key, (), 0, env.num_actions))  # device op + BLOCKING host sync
```
`int(device_array)` forces a device->host transfer and blocks async dispatch every
step. DreamerV3 avoids on-device RNG for random exploration entirely: its
`RandomAgent.policy` samples with **NumPy** (`np.stack([v.sample() for _ in range(batch_size)])`,
danijar/embodied `embodied/core/random.py`). Mirror that:
```python
rng = np.random.default_rng(seed)
actions = rng.integers(0, env.num_actions, size=prefill_steps)  # host, vectorized, once
for i in range(prefill_steps):
    frame = env.step(int(actions[i]))   # int() on a numpy scalar is free (no device sync)
```
This removes N `jax.random.split` + N `randint` device dispatches + N blocking
syncs from the hot path. (For a random uniform action, on-device RNG buys nothing
— reproducibility is preserved by seeding NumPy.)

### 3b. Persistent compilation cache (survive process restarts)
Current API ([Persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)):
```python
jax.config.update("jax_compilation_cache_dir", "/persistent/path/jax_cache")  # NOT /tmp
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # cache everything, no size floor
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # cache even fast compiles
```
Defaults: `min_compile_time_secs = 1.0` (graphs compiling faster than 1 s are NOT
persisted), `min_entry_size_bytes = 0`. Env-var equivalent:
`JAX_COMPILATION_CACHE_DIR`. The extractor already sets
`os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/vggt_jax_cache")`
(`feature_extractor.py:26`) — change the target off `/tmp` (node-local, wiped) to
shared storage, and lower `min_compile_time_secs` to 0 so the sub-second head/agg
graphs are actually written. Note: persistent cache only removes **XLA compile**
time on restart; it does not remove per-step tracing or dispatch. It will not
speed up steady-state prefill, only the first ~warmup.

### 3c. AOT compile (optional, removes first-call tracing jitter)
`compiled = jax.jit(f).lower(*sample_args).compile()` then call `compiled(*args)`.
Gives a `Compiled` object you can call directly; also `compiled.cost_analysis()`.
([jax.jit page](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html), AOT
section of docs.) The extractor instead does eager warmup with representative
frames in `__init__` (`_warmup`, `feature_extractor.py:480-528`), which achieves
the same "pay compile once up front" goal.

### 3d. Kill remaining per-step host syncs
Any of `int(x)`, `float(x)`, `x.item()`, `np.asarray(device_array)`,
`bool(x)`, printing a device array, or a Python `if x > 0:` on a device scalar
forces a blocking transfer and serializes the async pipeline. In the hot path:
- `run_multi_episode.py:44` `int(jax.random.randint(...))` — fixed by 3a.
- `feature_extractor.py:791` `int(jnp.asarray(valid_len))` per camera block/frame
  (only when `compute_heads=True` and camera cache full).
- `feature_extractor.py:826` `int(jnp.asarray(patch_start_idx))` per frame — value
  is always `1 + num_register_tokens = 5` (comment `feature_extractor.py:820`);
  cache it as a Python constant instead of re-syncing.
- `replay_buffer.add` / `transition_from_fields` — verify they do not call
  `np.asarray`/`.item()` on live device arrays each step (device->host copy). If
  the adapter emits device arrays and the NumPy buffer needs host arrays, one copy
  is unavoidable, but it should be a single explicit `jax.device_get` at the end,
  not scattered scalar syncs.

### 3e. `block_until_ready` only where needed
Use `block_until_ready` only to (a) delimit a profiling window or (b) prevent
unbounded async queue growth. The extractor correctly restricts it to warmup
(`feature_extractor.py:494,509,518,526`) and profiling
(`feature_extractor.py:870-872`). Do NOT add it to the steady-state prefill path —
it defeats async dispatch overlap. If the buffer `add` already pulls arrays to
host (a natural sync point), no extra barrier is needed.

## 4. DreamerV3 / embodied prefill patterns (primary source)

- **Random exploration is NumPy, not jitted, not on-device.** `RandomAgent.policy`
  returns `{k: np.stack([v.sample() for _ in range(batch_size)]) ...}` — pure host
  sampling, batched over envs (danijar/embodied `embodied/core/random.py`). Confirms
  fix 3a: there is no benefit to `jax.random` for uniform random prefill actions.
- **The env-interaction loop is a Python loop, not `lax.scan`.** DreamerV3 drives
  envs through `embodied.run.train(...)` with a `Driver` (danijar/dreamerv3
  `dreamerv3/main.py`, `embodied.run.train`). The env step boundary is inherently
  Python (Habitat/Gym envs are stateful host objects and cannot be traced), so
  `lax.scan` is not applicable to the outer loop — only the *agent's* pure JAX
  policy/train step is jitted. This matches the repo design: jit the VGGT extract,
  keep the env loop in Python.
- **Async dispatch:** because the policy is jitted and returns device arrays, and
  actions are consumed by the (host) env, the only mandatory sync is converting the
  chosen action to a host scalar. DreamerV3 keeps that single conversion and lets
  everything upstream stay async. Same principle applies here: one host sync per
  step maximum, not several.
- Config guidance: DreamerV3 README documents increasing exploration during
  prefill; the prefill data-collection phase uses the random agent before world-model
  training. (danijar/dreamerv3 `README.md`.)

## 5. Is Habitat, not JAX, the bottleneck? (how to prove it)

Almost certainly yes for a large share of the step. Repo memory
`house-points-pose-step-cost` already measured production ~219 ms/step = VGGT
forward 132 ms + replay `sample` 59 ms (amortized) + overhead, on a step that does
NOT even include `env.step` render cost. Recompilation (a one-time cost of ~seconds
during warmup) cannot explain a *steady* per-step time; if steps are uniformly slow
(not a few slow then fast), recompilation is ruled out by construction.

To attribute cost, distinguish two layers:
- **Coarse wall-clock, per section** (cheapest, do this first). Wrap each phase in
  `time.perf_counter()`: `env.reset/step`, `feature_extractor.extract`,
  `adapter_fn`, `transition_from_fields`, `replay_buffer.add`. IMPORTANT: to time
  the JAX `extract` truthfully you must `block_until_ready()` on its outputs at the
  timer boundary, otherwise async dispatch hides the real cost in the next sync.
  Habitat `env.step` + sensor render is fully synchronous CPU/GPU host work and its
  `perf_counter` delta is already truthful.
- **Device timeline** (`jax.profiler.trace`). Wrap ~20 steps:
  ```python
  with jax.profiler.trace("/persistent/jax-trace", create_perfetto_trace=True):
      ... run ~20 prefill steps ...
  ```
  `create_perfetto_trace=True` dumps `perfetto_trace.json.gz` for
  https://ui.perfetto.dev; it captures CPU + GPU + Python + on-device ops on one
  timeline. Gaps between XLA ops == host-bound time (Habitat render, NumPy buffer,
  Python overhead); back-to-back dense XLA ops == compute-bound (VGGT forward).
  Recompiles show up as long one-off XLA compilation spans. ([Profiling computation](https://docs.jax.dev/en/latest/profiling.html), [jax.profiler.trace](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html))

Decision rule: if the Perfetto timeline shows the GPU idle while Habitat renders,
optimize the env (async/vectorized envs, fewer sensors, lower render res), not JAX.

---

## Sources

Repo code:
- `prototyp/house_context_multi_episode/run_multi_episode.py:36-59` (prefill loop; per-step `int(jax.random.randint)` at :43-44)
- `src/vggt/jax/feature_extractor.py:26` (compilation-cache env var), `:159,282,436-448,580-581` (padded (k,v,valid_len) fixed-shape cache), `:197,224,242` (module-scope jits + static_argnums), `:377-378,716-720` (default fixed budgets), `:733` (is_first_frame bool -> 2 compiles), `:480-528` (warmup block_until_ready), `:791,826` (per-frame host syncs), `:870-872` (profiling-only sync)
- Repo memory: `house-points-pose-step-cost.md` (~219 ms/step attribution)

JAX docs (docs.jax.dev):
- [Debugging slow JAX tracing and compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) — log_compiles, explain_cache_misses, DUMP_IR, recompile causes
- [Configuration options](https://docs.jax.dev/en/latest/config_options.html) — jax_explain_cache_misses, jax_log_compiles semantics
- [Persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) — jax_compilation_cache_dir, min_compile_time_secs=1.0 default, min_entry_size_bytes
- [Profiling computation](https://docs.jax.dev/en/latest/profiling.html) and [jax.profiler.trace](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html) — create_perfetto_trace
- [jax.jit](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) — lower().compile() AOT

DreamerV3 / embodied (github.com/danijar):
- danijar/embodied `embodied/core/random.py` — RandomAgent samples actions with NumPy, batched
- danijar/dreamerv3 `dreamerv3/main.py` (RandomAgent wiring, `embodied.run.train`) and `README.md` (prefill/exploration)

## Open questions

- **`jitted_fn._cache_size()`** is used in practice and lives in `jax/_src/pjit.py`,
  but I could not confirm it on a public docs page — treat as internal API that may
  change between JAX versions. Verify against the installed JAX version before
  relying on it in a committed diagnostic.
- Exact `_calculate_dynamic_budgets` behaviour: whether any config path in this
  repo ships with dynamic (non-override) budgets that would make the aggregator
  `static_argnums` tuple vary per frame. Default constructor is safe; confirm no
  YAML/config sets `budgets_static` to a per-frame-varying source.
- Whether `replay_buffer.add` / `transition_from_fields` / `HouseContextAdapter`
  perform a `np.asarray`/`device_get` on live device arrays each step (I did not
  read those files) — this is a probable second per-step host sync worth auditing.
- Habitat env: whether it supports async/vectorized stepping to overlap render with
  the VGGT forward; not covered by the sources consulted.
