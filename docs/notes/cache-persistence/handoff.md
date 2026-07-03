# Handoff: persisting the VGGT attention cache across episodes

> Canonical copy. The experiment plan lives in
> `src/prototyp/cache_persistence/EXPERIMENTS.md` (kept with the future
> experiment drivers, per `src/prototyp/AGENTS.md`). A source-file reference
> index lives alongside this file at `references.md`.
>
> Status: **partially implemented** (2026-07-02). §2 budget raised to the
> InfiniteVGGT default 1.2M tokens; §4.2b camera-head sliding-window eviction
> landed; §4.3 save/load/reset-for-scene API landed (`ResetMode` enum). The
> live reset path and adapter wiring (§3, §7 step 3) are **not yet** flipped —
> default behaviour is unchanged. Read together with
> `docs/notes/house-context-episode-reanchoring.md` (the misalignment fix,
> Option C) and `docs/notes/house-context-misalignment-problem.md` (the
> plain-language bug). This document is about a **different** lever — the
> **VGGT KV cache**, not the point buffer.

## 0. The problem, in one sentence

`JAXVGGTFeatureExtractor.reset()` fires at every episode boundary and
**throws away the entire streaming attention state** (`_past_kvs_padded`,
`_last_scores`, `_past_kvs_camera`, `_frame_idx`), so VGGT re-anchors its
world frame to episode-1's first camera and re-derives geometry from
scratch — the live house context never benefits from *attention*
accumulated in previous episodes.

## 1. Two caches — do not confuse them

This is the single most important thing to internalise before touching code.
The "house context" stack has **two independent stores**, and `reset()` only
touches one:

| Store | What it holds | Lives in | Cleared by `extractor.reset()`? | Bounded? |
|---|---|---|---|---|
| **VGGT KV cache** (`_past_kvs_padded`, `_past_kvs_camera`, `_last_scores`) | Streaming attention K/V for the aggregator + camera-head; drives per-frame reconstruction quality via temporal attention | `JAXVGGTFeatureExtractor` instance state | **Yes — fully cleared** | Yes, bounded (see §2) |
| **`HouseContextPoseBuffer`** | The persistent, voxel-deduped 3D **house map** the encoder consumes | one per `scene_id` inside `VGGTHousePointsPoseObsAdapter` (`hybrid_adapter.py`) | **No** — survives episodes of the same scene (by design) | Yes (`HOUSE_CONTEXT_MAX_POINTS`) |

The misalignment bug (other two notes) is about the *buffer* mixing
coordinate frames across episodes. **This handoff is about the VGGT KV
cache** — the attention state that `reset()` wipes. "Save the cache for the
house context" means: make the VGGT *attention* state survive episode
boundaries (and possibly across agents), so the per-frame VGGT outputs
benefit from attention accumulated over prior episodes of the same house.

## 2. How the VGGT cache is actually bounded (not unbounded)

"Never reset" was rejected in the reanchoring note as "unbounded KV cache."
That is only half true — the **aggregator** cache self-bounds via eviction;
the **camera-head** cache did not (until 2026-07-02), and was the real blocker.

### Aggregator cache (self-bounded, evicts)

- `Aggregator` depth = 24 blocks. Each block keeps a K/V cache.
- Per-call budget: `current_budgets[b]` tokens per block. In **training** the
  encoder now uses `VGGT_TOTAL_BUDGET = 1_200_000`, `VGGT_STATIC_BUDGETS =
  (50_000,)*24` (`encoders/base.py:147-148`, raised 2026-07-02 from the
  200k/8333 override to the InfiniteVGGT default for a fair token-space
  comparison) — so each block holds ≤ 50 000 tokens. The extractor default is
  `_DEFAULT_TOTAL_BUDGET = 1_200_000` (uniform ≈ 50 000/block); both now agree.
- Tokens per frame per block: `P = 1 + num_register_tokens + patch_grid² =
  1 + 4 + 37² = 1374`. At the 1.2M budget each block keeps the permanent
  **anchor frame** (1374 tokens) + ≈ (50_000−1374)/1374 ≈ **35 recent diverse
  frames** (~36-frame streaming window). (At the old 200k budget it was ~6.)
- **Eviction policy** (`attention.py:_padded_evict` / `_evict_kv`): when a
  block overflows its budget it keeps the first `num_anchor_tokens = P`
  tokens as permanent **anchors** (frame-0 tokens, never evicted) and, among
  the candidate tokens, **retains the lowest cosine-similarity-to-mean** —
  i.e. the most *diverse/unique* tokens. Redundant tokens (similar to the
  cache mean) are pruned. `last_scores` feeds `_calculate_dynamic_budgets`
  (softmax over `2·(1−score)`), which **shrinks** the budget of blocks that
  stored high-similarity (low-diversity) tokens last frame — a feedback loop
  that pushes budget toward diverse blocks.

  **Consequence for "remember the sofa":** a re-observed sofa produces
  tokens highly similar to the existing sofa tokens → they are *redundant*
  → **evicted**. The cache keeps *one* representative of the sofa (the anchor
  frame if the sofa was in frame 0, otherwise the first diverse observation)
  and **discards duplicates**. It dedups; it does not accumulate or improve
  the sofa across re-observations. The anchors are **frame-0 specific**, so
  if agent B starts in a different room, its anchors are a *different* room's
  geometry — there is no shared cross-agent cache today.

### Camera-head cache (fixed window, **now sliding** — was: raised on overflow)

- `_CAM_MAX = max_camera_frames * _cam_num_iters`; default
  `max_camera_frames = 1024`. The camera head has **no internal eviction**
  (its trunk blocks run with `cache_budget=None`, unlike the aggregator).
  Until 2026-07-02 the extractor's `_check_camera_cache_capacity` **raised
  `RuntimeError`** at frame `max_camera_frames` — the hard blocker for "no
  reset, run 1 M steps."
- **✅ Resolved (2026-07-02, §4.2b):** the capacity check now calls
  `_evict_oldest_camera_frame` — a sliding window dropping one frame per new
  frame once full. The camera head keeps the most recent `max_camera_frames`
  frames (InfiniteVGGT semantics). The aggregator always survived (eviction);
  the camera head now does too.

  **Two caches, two horizons — do not conflate:** the aggregator is bounded by
  *token budget* (~36 frames/block, diversity-evicted, keeps diverse
  representatives across the whole stream); the camera head is bounded by
  *frame count* (1024 frames, sliding, most-recent only). The camera head is
  a *recent-context* window for pose refinement, not a long-term memory —
  long-horizon house-context memory lives in the aggregator's diversity
  cache + the `HouseContextPoseBuffer`, not the camera head. Raising
  `max_camera_frames` lengthens the camera window at linear memory cost; it
  cannot cover 1 M frames, which is why the sliding window is the viable path.

## 3. Where `reset()` is wired in

- `feature_extractor.py:_image_from_extract_input` — when an
  `ObservationFrame` has `is_first=True`, calls `self.reset()` before
  extracting. This is the live path: every episode's first frame resets.
- `adapters/vggt_adapter.py:50` and `adapters/hybrid_adapter.py` lines 95,
  134, 221, 408 — adapters register `on_episode_reset=extractor.reset`
  (some paths pass `None`, e.g. the context-Transformer path at line 134/221
  deliberately keeps the extractor live across resets).
- `reset()` itself (`feature_extractor.py`) clears all four fields.

So there are already two regimes in the codebase: the per-episode-reset
regime (most adapters) and the **never-reset regime** (the context-Transformer
adapter, `on_episode_reset=None`). The "no reset in level 1" request is
closer to the latter. The camera-head overflow that used to block this is now
resolved (§4.2b, 2026-07-02); the remaining work is the per-scene restore
wiring (§7 step 3) and the experiments (§7 steps 4–5).

## 4. The decision to make (this is the design question)

We want, for the **live house context in curriculum level 1**, the VGGT
attention state to **persist across episodes of the same scene** (and the
experiments in `src/prototyp/cache_persistence/EXPERIMENTS.md` ask whether it
should persist across agents too). Concretely, three sub-decisions, each with
a recommendation:

### 4.1 Aggregator cache: persist per-scene, key by scene_id

- Stop calling `reset()` at episode boundaries for level 1; instead,
  **save/restore** the aggregator cache keyed by `scene_id`, exactly like
  the `HouseContextPoseBuffer` is already keyed per scene
  (`hybrid_adapter.py` owns one buffer per `scene_id`).
- Mechanism: a `scene_cache_store: dict[scene_id, AggregatorCacheSnapshot]`
  (already landed on the extractor, §4.3). On episode start,
  `reset_for_scene(scene_id)` (restore if known, fresh if unseen); never call
  the global `reset()`. The eviction policy already bounds it — memory is
  O(num_scenes × budget), not O(steps).
- This is **Option D done correctly**: not "one VGGT frame forever" (which
  teleports across scenes), but "one VGGT stream per scene, resumed each
  episode of that scene." Teleports within a scene between episodes remain
  (Habitat randomises the start pose), so the temporal-attention assumption
  is still violated at the resume point — see risk R2.

### 4.2 Camera-head cache: add eviction or a per-scene cap

The camera head had no eviction and raised at 1024 frames. For a per-scene
stream that may run hundreds–thousands of frames total across episodes, two
options:

- **(a) Raise `max_camera_frames`** to cover the longest expected per-scene
  frame count and accept the linear camera-cache memory cost. Simple, but
  memory = `max_camera_frames × num_iters × H × head_dim × 2 (k,v) × scenes`.
  At 1024 frames × `_cam_num_iters` this is already large; 1 M frames is
  infeasible this way.
- **(b) Add a sliding-window eviction to the camera head** mirroring the
  aggregator: keep the last N camera iterations, drop the oldest. This
  needs a code change in the camera-head apply path. Preferred for any
  long-run experiment; the camera pose is most influenced by recent frames
  anyway. **✅ DONE (2026-07-02):** `_check_camera_cache_capacity`
  now calls `_evict_oldest_camera_frame` (shift each trunk block left by
  `_cam_num_iters` rows, pad the tail, set `valid_len = _CAM_MAX − n`) instead
  of raising. Output is byte-identical for the first `max_camera_frames`
  frames; past that the camera head keeps a sliding window of the most
  recent `max_camera_frames` frames (InfiniteVGGT semantics). Cost: one
  concat per trunk block per frame once full (R5) — watch the wall-clock
  regression in E0/E2.

### 4.3 "Load the attention" on episode start — the save/restore API

"The episode which the house context should load the attention" = when an
episode of scene S starts, the extractor loads scene S's saved aggregator
+ (evicted) camera cache instead of starting empty. Design surface:

```python
# Landed next to reset() in feature_extractor.py
class ResetMode(Enum):              # FULL (default) | PERSIST_SCENE
class AggregatorCacheSnapshot: ...  # frozen holder for the 4 streaming fields
def save_cache(self) -> AggregatorCacheSnapshot: ...   # refs, no device copy
def load_cache(self, snap) -> None: ...
def reset_for_scene(self, scene_id) -> None:          # FULL->reset(); PERSIST->save/restore
```

**✅ DONE (2026-07-02):** `ResetMode` enum (`FULL` default / `PERSIST_SCENE`),
`AggregatorCacheSnapshot` frozen dataclass, and `save_cache` / `load_cache` /
`reset_for_scene` landed in `feature_extractor.py`. `save_cache` holds JAX
array references (immutable, no device-memory copy); `reset_for_scene` under
`FULL` is identical to `reset()` (default behaviour unchanged). The adapter
wiring (step 3) is **not** done — experiments must call the API directly for
now.

Adapter side: `VGGTHousePointsPoseObsAdapter` already keys a buffer per
`scene_id`; add a parallel `dict[scene_id, snapshot]` and call
`extractor.reset_for_scene(scene_id)` at episode start instead of
`extractor.reset()`. `on_episode_reset` becomes a per-scene restore, not a
wipe.

## 5. Risks (these are what the experiments must measure)

- **R1 — Attention drift over a long stream.** VGGT was trained on
  sequences; a 1 M-frame stream is far out of distribution. The diversity
  eviction may collapse to a stale anchor + a few tokens, or attention
  quality may drift. *Measure:* per-frame `world_points`/`camera_pose` drift
  vs. a fresh-reset baseline (the existing `VGGTCacheDiagnostic` already
  plots point-delta, camera-L2, global-token cosine — extend it to long
  horizons, EXPERIMENTS E2).
- **R2 — Teleport at episode resume.** Even with per-scene persistence,
  episode k+1 starts at a *different pose* than episode k ended. The
  aggregator assumes a temporally coherent camera path; a teleport injects a
  discontinuity. *Measure:* the first few frames after a resume — does
  VGGT's pose estimate spike / does the point map tear? Compare to the
  reanchoring approach (Option C) which handles this with GT-pose Umeyama.
- **R3 — Anchor-frame dominance.** The permanent anchors are frame-0 of the
  *scene's first episode ever*. If that frame is unrepresentative (a blank
  wall), a low-quality anchor set is locked in forever for that scene.
  *Mitigation:* refresh anchors when a higher-quality frame is seen, or drop
  the anchor-permanence and rely purely on diversity eviction — needs a code
  change and its own ablation.
- **R4 — Cross-agent sharing is not free.** Agents A and B exploring the
  same scene start at different poses → different anchor frames → different
  cache states. "Does the cache remember the sofa across agents?" is really
  "can agent B *load* agent A's saved cache and still benefit?" The
  frame-0-anchor mismatch and the teleport (R2) both apply. *Measure:*
  EXPERIMENTS E3.
- **R5 — Memory / cost.** Per-scene persistence = O(num_scenes × budget) for
  the aggregator; with the level-1 curriculum this is bounded by the number
  of L1 scenes (small). The camera head is bounded by the sliding window
  (§4.2b) but pays one concat per trunk block per frame once full — measure
  the wall-clock regression vs. the 158 ms/step baseline (EXPERIMENTS E0/E2).

## 6. What exists to build on (do not reinvent)

- `src/prototype_helpers/vggt_cache_diagnostics.py` — `VGGTCacheDiagnostic`,
  `read_cache_state`, `compare_features`, `plot_cache_drift`. Already logs
  aggregator/camera `valid_len`, point-delta percentiles, camera L2,
  global-token cosine. **Extend this**, do not write a parallel harness.
- `src/prototyp/prototype_vggt_cache_diagnostic.py` — the manual driver;
  `run_same_image_diagnostic` already compares fresh-reset vs. cached
  same-image drift over `NUM_CACHE_STEPS=20`. The long-horizon and
  cross-episode experiments generalise this.
- `src/vggt/jax/feature_extractor.py` — `reset`, `_past_kvs` property,
  `_check_camera_cache_capacity`, `_evict_oldest_camera_frame`,
  `save_cache`/`load_cache`/`reset_for_scene`, `ResetMode`,
  `AggregatorCacheSnapshot`. See `references.md` for exact locations.
- `src/r2dreamer/adapters/hybrid_adapter.py` (the `VGGTHousePointsPoseObsAdapter`,
  ~line 351+) — where per-scene restore hooks in alongside the per-scene
  buffer.

## 7. Sequence to implement (after experiments inform the design)

1. ✅ **DONE (2026-07-02):** Land `save_cache` / `load_cache` /
   `reset_for_scene` + `ResetMode` enum on the extractor (pure state copy;
   no behaviour change — default `ResetMode.FULL`).
2. ✅ **DONE (2026-07-02):** Add camera-head eviction (§4.2b) — required
   before any long run. Sliding-window evictor replaces the `RuntimeError`.
3. ⬜ Wire per-scene restore into `VGGTHousePointsPoseObsAdapter` behind a
   flag; default off so current training is unchanged.
4. ⬜ Run E1–E3 (`src/prototyp/cache_persistence/EXPERIMENTS.md`) to decide:
   persist per-scene only, or also cross-agent; keep anchors permanent or
   refresh.
5. ⬜ Only then flip the flag for level-1 training and re-run the 1 M-step
   smoke with the `afterany` chain to 2 M (see memory: house-points-pose
   step cost ≈ 158 ms/step, 1 M ≈ 44 h, fits one H100 job).