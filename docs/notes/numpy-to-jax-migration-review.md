# NumPy → JAX Migration Review (`src/`)

**Date:** 2026-07-03 · **Method:** 3 parallel read-only audit agents covering
(1) `r2dreamer` + `buffer` + `shared` + `main.py`, (2) `vggt` + `environments` +
`baselines` + `configs`, (3) `prototyp` + `prototype_helpers` + `analysis`.
Review only — no code was modified.

## TL;DR

The codebase is already in very good shape relative to the "prefer `jax.numpy`"
rule. Across all of `src/`, **36 files touch NumPy, and ~32 of them use it
exactly where the project's own carve-outs say they should** (host I/O,
simulator/torch/matplotlib interop, host RNG, voxel-key int64 math). Every
compute hot path — the VGGT JAX port, the live inference loop, the voxel
dedup/insert path in `HouseContextPoseBuffer`, the agent/trainer math — is
already pure `jnp`.

There is **no wholesale migration to do**. The actionable work is small and
specific: eliminate a handful of per-frame `jnp → np → jnp` round trips in the
adapter layer, and align two agent-obs dtypes with the bfloat16 preference.

## Actionable recommendations (priority order)

### P1 — `src/r2dreamer/adapters/hybrid_adapter.py`: kill per-frame device round trips
Lines ~101–111, ~141–160, ~262–296. Pattern today (runs **every env step**):

```python
wp_cp_host = np.asarray(wp_cp, np.float32)        # device → host (needed for replay)
replay["wp_cp"] = wp_cp_host.astype(np.float16)
agent_obs = jnp.asarray(replay["wp_cp"], ...)     # host → device again  ← wasted
```

The host copy for the replay buffer is legitimate (replay stores host NumPy).
The waste is rebuilding the **agent obs** from the host copy. Derive it from
the original device tensor instead:

```python
agent_obs = wp_cp.astype(jnp.bfloat16)            # stays on device
replay["wp_cp"] = np.asarray(wp_cp, np.float16)   # single device→host copy
```

Same fix applies to the token readout (`_extract_tokens`) and the projected
house context (`_project_context` — consider caching `_context` as a jnp array
so both conversions vanish; the replay path still takes one host copy).
**This is the single biggest transfer-churn win in the repo.**

### P2 — `src/r2dreamer/observation_preparation/vggt_readouts.py`: dtype only
Lines ~226, ~251. Structure is already ideal (one host copy for replay, agent
obs stays on device). Only change: agent-side `.astype(jnp.float16)` /
`.astype(jnp.float32)` → `jnp.bfloat16` per project preference. Verify the
downstream encoder accepts bfloat16 before switching.

### P3 — `src/buffer/house_context_pose_buffer.py:485`: fold host transpose
`np.moveaxis(observation.image, 0, -1)` then `jnp.asarray(...)` per frame.
Micro-opt: transfer once and transpose on device
(`jnp.asarray(observation.image).transpose(1, 2, 0)`). Keep `uint8` (raw RGB,
not bfloat16).

### P4 (optional, low value) — diagnostics on-device reductions
- `src/prototype_helpers/vggt_cache_diagnostics.py` (~L357–420): divergence
  math (`np.linalg.norm`, `np.percentile`, cosine) pulls full `(N,3)`
  world-point / token tensors to host per comparison. Could compute reductions
  on device and pull back scalars only. Debug harness, not production — do
  only if diagnostic runtime ever matters.
- `src/prototyp/graph_house_context/exp3_gft_compress.py:148`:
  `np.asarray(basis @ truncated)` inside the per-block loop copies a `(B,3)`
  array to host just to compute a scalar MSE — reduce on device
  (`float(jnp.sum(...))`) instead. **`src/prototyp/` is read-only by
  convention; needs explicit approval before touching.** Offline experiment,
  low urgency.

## What should NOT be converted (and why)

| Area | Files | Reason |
|---|---|---|
| VGGT host↔device glue | `vggt/jax/feature_extractor.py` | `int(np.asarray(...))` scalar pulls & `np.clip` budgets produce concrete Python ints for JIT `static_argnums` — structurally required. The existing `_budgets_static_override` already avoids the per-frame budget sync when budgets are fixed. |
| Checkpoint / weight I/O | `vggt/jax/weight_transfer.py`, `r2dreamer/checkpointing.py` | One-time torch↔Flax layout conversion and pickle serialization; host-only by definition. |
| Env interop | `environments/{habitat,crafter,observation}.py` | Habitat/gym return concrete NumPy; `ObservationFrame.image: np.ndarray` is the deliberate host observation contract. Converting would force premature device placement of every raw frame. |
| Replay sampling RNG | `buffer/replay_buffer.py:225` | Host RNG in a non-jit Python-loop gather path — the allowed carve-out. (Side note: it's unseeded by the project's JAX key stream — a determinism concern, not a JAX one.) |
| Overflow-safe index math | `buffer/house_context_pose_buffer.py:470` | int64 host math avoids int32 overflow (`point_count * max_points > 2^31`); JAX defaults to int32. |
| Static house-context encode | `r2dreamer/observation_preparation/static_house_context.py` | PLY parsing (`np.loadtxt`), dynamic-shape finite-row masking, `np.ravel_multi_index` (no jnp equivalent), `np.add.at` scatter-add — once-per-scene host code, correctly float32/int64 voxel-key math. |
| Ragged voxel-block grouping | `prototype_helpers/graph_ops.py` (`group_indices_by_block`) | Produces a variable-length list of index arrays — inherently non-jittable dynamic shapes; the rest of the file is already JAX. |
| Plotting / video / PLY | `prototype_helpers/{ply_io,observation_video,point_change_plot}.py`, `shared/video_utils.py`, prototyp exp1/2/4 | matplotlib / moviepy / PIL need concrete contiguous NumPy; textbook I/O boundary. |
| Deliberately JAX-free analysis | `analysis/invariance_metrics.py` | Docstring: "No torch/jax — safe to run on the login node." Uses float64 SVD/Umeyama for precision. Converting violates its design goal. |
| Metrics / profiling stats | `baselines/random_agent.py`, `shared/wandb_utils.py`, `vggt/jax/{profile,benchmark}_streaming.py`, `launch/evaluate.py`, trainer `action_counts` | Aggregation over Python scalar lists / tiny host counters for logging. |

## Structural findings

- **`src/vggt/` is fully project-owned.** The vendored upstream PyTorch model
  lives at `external/InfiniteVGGT/src` (see `vggt/paths.py`); `vggt/jax/` is
  the JAX port, `vggt/reference/` a torch parity wrapper. No "don't-touch"
  subtree inside `src/`.
- **`src/configs/` and `src/main.py` are NumPy-free.**
- **No bfloat16 violations found.** The `jnp` compute path uses bfloat16 for
  activations and float32 only where numerics demand it (softmax, masks, RoPE
  tables, score/budget and voxel-key math) — matching the stated policy. The
  only dtype gaps are the two agent-obs casts in P2.
- **Unavoidable per-frame device syncs to be aware of** (not fixable, just
  budget for them): the JIT static-arg scalar pulls in
  `feature_extractor.py`, and jnp→np conversions at the recording/plotting
  boundary (`observation_video`, `point_change_plot`) — recording frequency
  drives transfer volume.

## Count summary

| Slice | Files w/ NumPy | Must-stay | Actionable |
|---|---|---|---|
| r2dreamer / buffer / shared / main | 17 | ~14 | `hybrid_adapter` (P1), `vggt_readouts` (P2), buffer moveaxis (P3) |
| vggt / environments / baselines / configs | 9 | 9 | none |
| prototyp / prototype_helpers / analysis | 10 | ~9 | 2 optional diagnostics (P4) |
| **Total** | **36** | **~32** | **3 real + 2 optional** |
