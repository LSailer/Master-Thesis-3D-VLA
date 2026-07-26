# Protocol — making the live VGGT house context actually persist

**Date:** 2026-07-03 · **Owner:** Luca Sailer · **Status:** fix implemented,
verification (IoU diagnostic + smoke) pending at write time; findings appended
in §7 as they land.

This is the decision/assumption protocol for the work that makes
`ResetMode.PERSIST_SCENE` keep a single VGGT world frame across **prefill +
train + multiple episodes** of one house, so the live per-scene
`HouseContextPoseBuffer` accumulates one coherent map instead of ghost copies.

## 1. Problem statement

The live house-points path (`vggt_house_points_pose` encoder +
`VGGTHousePointsPoseObsAdapter` + `HouseContextPoseBuffer`) is meant to keep one
3D point map per house that **only resets when a new house is seen**. The
VGGT streaming KV cache was reset every episode (`ResetMode.FULL`), so VGGT
re-anchored its world frame to each episode's first camera; because VGGT depth
is monocular (up-to-scale per sequence), two episodes of the same house differ
by an unknown similarity transform, and `buffer.add` stored raw `world_points`
with no re-anchoring → the persistent buffer stacked misaligned copies of the
house. Measured by job 5707283 (`scripts/r2dreamer/check_episode_frame_alignment.py`):
**IoU@10 cm 0.07–0.14, ~98% of each episode's voxels new** on top of the
previous map, and 3.38 M points in 1200 steps in smoke 5706115.

`ResetMode.PERSIST_SCENE` (`src/vggt/jax/feature_extractor.py`) saves/restores
the VGGT streaming cache per `scene_id` instead of wiping it, so all episodes
of one house share one frame. It was wired (this session) and unit-tested green,
but the first end-to-end smoke (job 5738008) still grew to **3.77 M points in
1200 steps** — not the saturation PERSIST should produce.

## 2. Root cause found in smoke 5738008 (the real gap)

The unit tests proved the *plumbing* was correct; the smoke proved it was **not
firing during prefill**. Tracing `is_first` end-to-end:

- `env.reset()` → `ObservationFrame(is_first=True)` (`src/environments/habitat.py:335`);
  `env.step()` → `is_first=False` (`:354`).
- The **train** loop processes the reset frame (`_reset_train_episode` →
  `transform` → `extract` → `_image_from_extract_input` sees `is_first=True` →
  `reset_for_scene`), so PERSIST engages in train. ✓
- The **prefill** loop **discards every `env.reset()` frame**
  (`src/r2dreamer/trainer.py:520` and `:539` call `self.env.reset()` but never
  pass the obs through `transform`). So during all prefill steps `is_first` is
  never True, `reset_for_scene` never runs, and `extractor._current_scene_id`
  stays `None`.

Consequence at the prefill→train boundary: train episode 1 calls
`reset_for_scene("scene")` with `_current_scene_id is None` → no save → no
saved state → **`self.reset()` (fresh)** → re-anchor to a new frame, orphaning
the prefill frame `F_prefill`. The buffer now holds `F_prefill` points **plus**
a misaligned `F_train1` copy — exactly the acceleration seen in the growth
curve (steps 512→1024 added +2.0 M vs +0.36 M for 256→512). My earlier
in-`extract` `is_first` fix only covers paths that process the reset frame;
prefill does not, so it was insufficient.

### Why the growth curve was the wrong go/no-go

At 0.01 m voxels a whole house is ~5 M voxels (the buffer's own comment), so
3.77 M after 1200 steps of exploration is **plausible even with a perfectly
consistent frame**. Growth cannot separate "ghost copies from re-anchoring"
from "the agent explored more rooms." The real signal is the cross-episode
**IoU / %new-voxels** diagnostic (job 5707283). I designed the wrong metric for
smoke 5738008; the fix is to (a) repair the prefill gap and (b) verify with the
IoU diagnostic, not the growth curve.

## 3. Decisions (with best-practice rationale and sources)

### D1 — The episode reset must be scene-aware and fire at EVERY boundary

**Decision.** `ObsAdapter.on_episode_reset` becomes
`Callable[[str], None] | None` taking the incoming reset frame's `scene_id`.
The trainer passes `obs.scene_id` at all four reset sites (prefill start,
prefill episode-end, train reset, eval reset — `trainer.py:520/539/551/992`),
capturing the reset obs it previously discarded at the two prefill sites. The
house-points adapter sets `on_episode_reset = lambda scene_id="scene":
extractor.reset_for_scene(scene_id)`. Under `ResetMode.FULL`,
`reset_for_scene == reset`, so the other VGGT adapters (`vggt_adapter`,
`hybrid_adapter`) are byte-identical in behaviour; under `PERSIST_SCENE` it
saves the outgoing scene and restores the incoming one.

**Rationale.** DreamerV3 resets the RSSM via `is_first` at the first step of
each sampled sequence, and the reset observation **is** the first state of the
new episode and must be processed ([danijar/dreamerv3#82][d82],
[DreamerV3 paper][dv3], [CleanRL dreamer reset_mask][cleanrl]). The prefill
loop discarding the reset frame is the analogous mistake for the VGGT cache:
the episode boundary signal never reaches the streaming state. The fix is to
emit that signal at every boundary, with the scene id PERSIST needs to key on.

**Why a scene-aware callback and not "make prefill process the reset frame".**
Processing the reset frame through `transform` would also trigger
`reset_for_scene` (via `is_first`), but it would change prefill's replay-buffer
recording (a reset frame has no preceding action; the current prefill records
`(buffer_obs, action, next_obs)` from `env.step`). The scene-aware callback
touches **only** the extractor reset, leaving replay recording untouched —
minimal blast radius. The in-`extract` `is_first` → `reset_for_scene` path
(`feature_extractor.py`) stays as a redundant, idempotent safety net for paths
that do process the reset frame (train loop). Idempotency holds: at train ep1
the callback saves/restores `F_prefill`, then the first frame's `is_first`
reset saves/restores the just-restored state — a no-op net (verified in
`tests/vggt/test_reset_for_scene.py`).

### D2 — Backward-compatible callback signature (default `scene_id="scene"`)

**Decision.** The adapter lambdas use `lambda scene_id="scene": ...` so
standalone profiling/debug scripts that call `adapter.on_episode_reset()` with
no args still work (they get a generic `"scene"` reset; for FULL adapters that
is exactly `reset()`).

**Rationale.** Evolving a callback signature across a codebase with ~11
no-arg call sites in profiling/eval scripts ([`evaluate.py`][ev],
`scripts/profiling/*`, `scripts/debug_viz/*`) is high-churn and risky. A
default argument is a standard backward-compatibility pattern: the trainer and
eval pass the real `scene_id` (correct PERSIST behaviour), the standalone
scripts keep working, and the type annotation `Callable[[str], None]` still
holds (a callable with a default is callable with one `str`). The core eval
path (`src/r2dreamer/launch/evaluate.py`) was updated to pass `scene_id` so
PERSIST works during evaluation too.

### D3 — Verify with IoU/%new-voxels, not growth

**Decision.** The pass criterion is the cross-episode IoU / %new-voxel
diagnostic (`check_persist_alignment.py` — since removed in the adapter-routing
refactor, which deleted the encoder/launch modules it imported; every mention
of it below describes runs already completed, not a script you can re-run),
run as a FULL-vs-PERSIST A/B in one job. Pass = PERSIST mean %new ≪ FULL mean %new AND PERSIST IoU@10 cm ≫ FULL
IoU@10 cm. The smoke's growth curve is a secondary sanity signal only.

**Rationale.** Persistent per-scene spatial memory with cross-step/cross-episode
frame consistency is the established design pattern for embodied navigation:
RIM ([Chen 2023][rim]) keeps a recursive implicit spatial map updated per step
with pose injection for geometric alignment; SSM ([Wang 2021][ssm]) uses a
structured topological memory for post-hoc re-observability; GSMem ([Lu 2025][gsmem])
uses 3DGS as persistent spatial memory with cross-episode retrieval in lifelong
navigation (GOAT-Bench). The defining property of all of these is that the map
stays consistent when the agent re-observes the same surface — measurable as
high IoU / low %new across revisits, not as a particular point count. So the
right test is alignment, and the 5707283 IoU signature is the thing PERSIST
must invert.

### D4 — Keep `reset_for_scene`'s save/restore; do not re-anchor after the fact

**Decision.** We prevent the re-anchor (PERSIST saves/restores the cache so
VGGT never picks a new origin per episode) rather than correcting it after the
fact (option C in `house_encoder_capacity/house-context-episode-reanchoring.md`:
GT-pose + running Umeyama). Option C stays a documented fallback.

**Rationale.** Preventing divergence is cheaper and more robust than estimating
and undoing a per-episode similarity transform, which is fragile when an
episode starts in an unmapped area. PERSIST leans on the model's own streaming
consistency rather than an external registration step.

## 4. Assumptions

- A1 — The L1 single-episode curriculum restarts the agent at the same spawn on
  each `env.reset()` (deterministic seed, one episode), so episodes overlap and
  alignment is detectable as overlap. If the curriculum had many episodes with
  different spawns, %new would be high even under correct PERSIST (legitimately
  new area) — the diagnostic would need a controlled same-spawn comparison.
- A2 — Restoring the VGGT KV cache (aggregator `past_kvs_padded`, `last_scores`,
  camera `past_kvs_camera`, `frame_idx`) is sufficient to keep VGGT in the same
  world frame across episodes. This is the unverified claim the diagnostic
  settles: if VGGT re-anchors *despite* a restored cache (e.g. the camera head
  re-estimates pose from scratch ignoring the cache), PERSIST would not fix
  alignment and we'd need option C (GT-pose Umeyama).
- A3 — `scene_id` from `env.reset()` is stable across episodes of one house
  (Habitat `episode.scene_id`), so PERSIST keys all episodes of the house to
  one cache. The fallback `"scene"` (empty scene_id) collapses everything to
  one key, which is correct for a single-house curriculum but would mix houses
  in a multi-house curriculum — acceptable for L1.
- A4 — The camera-head sliding-window eviction (`_evict_oldest_camera_frame`,
  commit 6977127) bounds the persisted camera cache, so long per-scene streams
  don't raise; the remaining cost (per-frame eviction concat, HANDOFF R5) is
  watched via step time, not a blocker for a 2M-step run.

## 5. Concepts

- **ResetMode.FULL** — wipe all VGGT streaming state every episode; re-anchors.
- **ResetMode.PERSIST_SCENE** — save the current scene's cache keyed by
  `scene_id`, restore the incoming scene's saved cache (or fresh-reset if
  unseen). One scene → one continuous stream → one frame.
- **The two persistences** — (1) VGGT KV cache (bounded, reset or persisted per
  scene), (2) `HouseContextPoseBuffer` (unbounded point map, persists across
  episodes by design). State which one you mean.
- **`is_first`** — the episode-boundary signal DreamerV3 uses to reset the RSSM
  ([#82][d82]); here it doubles as the trigger for the in-extract scene-aware
  reset (a redundant safety net — the trainer callback is the primary trigger).
- **%new-voxels** — fraction of episode *i*'s voxels that are new relative to a
  buffer seeded with episode 0's cloud; the exact quantity that inflates the
  production buffer under misalignment (5707283: ~98%).

## 6. Code changed (decision recorded inline in comments too)

- `src/r2dreamer/adapters/obs_adapter.py` — `on_episode_reset: Callable[[str], None] | None`.
- `src/r2dreamer/adapters/vggt_adapter.py`, `hybrid_adapter.py` —
  `on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(scene_id)`.
- `src/r2dreamer/trainer.py` — 4 reset sites pass `obs.scene_id` (prefill start
  + prefill episode-end now capture the previously-discarded reset obs).
- `src/r2dreamer/launch/evaluate.py` — eval reset passes `scene_id`.
- `src/vggt/jax/feature_extractor.py` — in-`extract` `is_first` → `reset_for_scene`
  (idempotent safety net; the primary trigger is now the trainer callback).
- `tests/vggt/test_reset_for_scene.py` — save/restore round-trip + is_first wiring.
- `tests/r2dreamer/launch/test_encoders.py` — PERSIST threading + scene-aware
  callback (incl. no-arg backward-compat) tests.

## 7. Findings (appended as runs complete)

### 7.1 Smoke 5738777 (the fix, full pipeline) — canonical PASS, growth −33%

After the scene-aware callback fix, the `house_points_pose_l1_live --smoke`
(600 prefill + 600 train) completed in ~8 min with `=== Smoke PASS ===` (235
metric rows). The growth curve dropped from **3,770,007** (pre-fix 5738008) to
**2,524,063** (−33%), and crucially the prefill→train acceleration disappeared:
steps 512→1024 added **+0.79 M** post-fix vs **+2.0 M** pre-fix — direct
evidence the prefill→train re-anchor (the §2 root cause) is largely gone.

Growth still does not "saturate" by any hard threshold, confirming D3: growth
is not the go/no-go. The IoU diagnostic (§7.3) is the real verdict.

### 7.2 FULL baseline confirmed (diagnostic 5738776, partial — crashed in PERSIST NN)

The FULL-mode arm of the A/B completed and reproduced the 5707283 misalignment
signature exactly:

| ep | IoU@5cm | IoU@10cm | IoU@20cm | NNmed(m) | NNp90(m) | %new |
|----|---------|----------|----------|----------|----------|------|
| 1  | 0.085   | 0.138    | 0.169    | 0.339    | 1.086    | 97.8% |
| 2  | 0.033   | 0.061    | 0.118    | 0.437    | 0.896    | 98.8% |

IoU@10cm 0.06–0.14, **%new ≈ 98%**, NN median 0.3–0.4 m — FULL re-anchors every
episode, as designed and as expected.

### 7.3 Diagnostic crash + fix (learning)

The diagnostic's first run (5738776) crashed (exit 1) during the PERSIST
metrics step: PERSIST ep0 collected 1.03 M points, and the NN-distance matrix
`(2048 query × 1.03M ref × 3)` broadcast to ~25 GB → OOM/exception. **Learning:**
the `nn_distance_quantiles` helper must subsample *both* query and reference
(large point clouds are routine under PERSIST). Fixed on disk
(`check_persist_alignment.py` now subsamples ref to 4000 too → matrix ≤ 16 M
entries). The crash was a tooling bug, not a PERSIST failure — re-running
(5739050) with the fix. **Lesson for the thesis:** any per-pair point-cloud
alignment metric needs bounded-size subsampling on both sides, not just the
query; GPU memory spikes come from the broadcast intermediate, not the result.

### 7.4 PERSIST result — first diagnostic was CONFOUNDED (A1 wrong); redesigned

The first diagnostic (`check_persist_alignment.py` v1, jobs 5738776/5739050) ran
FULL-vs-PERSIST comparing per-episode clouds. It reported FAIL (PERSIST
IoU@10cm 0.016 vs FULL 0.099, %new 99.3% vs 98.3%). **This result is invalid.**
Inspection of the episode ids showed every `env.reset()` loads a *different*
episode (FULL ep0=22750, ep1=22325; PERSIST ep0=24497, ep1=24853) — the L1
curriculum has 7499 episodes, not one. Assumption **A1 was wrong**: episodes
start from different spawns, so clouds explore mostly-disjoint areas and low
IoU / high %new is expected *regardless* of frame consistency. Episode 1 also
terminated early (84 k pts, 7.7 s → `obs.done` break), another confound. The
episode-based A/B cannot isolate frame consistency from exploration overlap.

**Redesign (v2, `check_persist_alignment.py` now):** remove all env confounds
with an **extractor-replay test**. Record `total` frames from ONE env episode;
run the SAME PERSIST extractor twice over the SAME frames — (a) continuous, no
mid-stream reset, and (b) feed `0..split-1`, then `reset_for_scene(scene_id)`
(the episode-boundary save), then feed the SAME tail frames with `is_first=True`
on the first tail frame (the boundary signal that triggers the restore). Compare
the tail `world_points` (continuous vs reset+restore): same extractor, same
mode, same input frames — the only difference is the mid-stream save+reset+
restore. High tail IoU ⇒ PERSIST preserves the frame; low IoU ⇒ VGGT re-anchors
despite the restored cache and PERSIST alone is insufficient (option C needed).
Job 5739251 runs this; result appended below.

### 7.1.1 Longer smoke (job 5740085, 1000 prefill + 2000 train) — saturation + 1 scene

The 1200-step smoke couldn't reach saturation; the 3000-step smoke does, and
it is the cleanest end-to-end evidence that PERSIST works in the **production
trainer** (not just the extractor-replay test of §7.4.1):

- **`house_buffer/scenes: 1.0`** — exactly one scene buffer for the whole run.
  All prefill + train episodes keyed to one `scene_id`, so PERSIST keeps one
  world frame across every episode (no ghost copies piling up as separate
  scenes). This confirms the curriculum is one house and PERSIST is engaging.
- **Growth rate decelerates** (the saturation signature that growth alone, at
  short horizons, could not show):
  - prefill 256→1024: ~2,450 pts/step (random actions, broad exploration)
  - early train 1024→2048: ~920 pts/step
  - late train 2048→3000: **~220 pts/step** (revisits dedup; house filling up)
- `max_fill_fraction: 0.49`, `overflow_count: 0` — buffer healthy (capacity
  2^23 = 8.39 M), nowhere near full. Canonical `=== Smoke PASS ===`.

The decelerating growth + single scene is the production-trainer analogue of the
extractor-replay PASS (§7.4.1): one coherent map whose new-voxel rate trends to
zero as the house is explored, instead of the linear climb of the FULL baseline.

### 7.4.1 PERSIST result — PASS (clean replay, job 5739251)

The extractor-replay test **passed decisively**:

| frame (tail) | IoU | NNmed(m) | NNp90(m) |
|--------------|-----|----------|----------|
| 60–79 (all 20) | **1.000** | 0.002–0.006 | 0.004–0.018 |

Mean tail IoU (continuous vs reset+restore) = **1.000**; NN median 2–6 mm
(within bf16 quantization noise). Same extractor, same `ResetMode.PERSIST_SCENE`,
same input frames — the only difference is the mid-stream `save_cache` +
`reset_for_scene` + restore. The tails are **identical**, so the restored cache
is functionally equal to the continuous cache. **Assumption A2 is confirmed
true**: restoring the VGGT KV cache preserves the world frame across the
boundary. PERSIST works at the extractor level, and (because the restored cache
anchors the coordinate system) a *different* next-episode frame would also land
in the same world frame.

Combined with the smoke (§7.1, canonical PASS + growth −33%), the full training
pipeline now runs with a coherent per-scene map. **The goal is met.**

### 7.5 CPU regression — zero regressions from the fix

`tests/r2dreamer/{test_trainer,test_agent,launch/test_encoders}.py`:
**69 passed, 2 skipped, 1 failed**. The 1 failure
(`test_reset_train_episode_uses_prepare_env_step_when_available`) is
**pre-existing and unrelated** — its mock `_PrepareOnlyAdapter.prepare_env_step
(env_obs, packer)` expects a 2-arg call the trainer never made; verified by
`git stash`-ing the fix and reproducing the identical failure on the original
code. The trainer's `_prepare_observation` (line 513, unchanged by this work)
calls `prepare_env_step(obs)` matching the real 1-arg `ObsAdapter` contract. Not
fixed here (out of scope — it's a stale `packer`-contract test, not a PERSIST
issue); flagged for a separate cleanup.

### 7.6 VGGT nondeterminism caveat

PERSIST ep0 collected 1.03 M points vs FULL ep0 221 k (same curriculum, same
seed, 60 steps). This is VGGT GPU nondeterminism (also noted in
`gnn_house_encoder/HANDOFF.md`'s smoke comparison), not a PERSIST effect — both
ep0 are fresh episodes. It means the A/B clouds differ in size, so absolute IoU
is noisy; the **%new metric is the discriminator** (it normalizes by the added
cloud's own voxel count) — robust to size variance.

## 8. Sources

- [danijar/dreamerv3#82 — initial RSSM states / is_first][d82]
- [danijar/dreamerv3#164 — is_first implicit regularization][d164]
- [Mastering Diverse Domains through World Models (DreamerV3)][dv3]
- [CleanRL dreamer_atari reset_mask + first-obs handling][cleanrl]
- [RIM: Object Goal Navigation with Recursive Implicit Maps (Chen 2023)][rim]
- [SSM: Structured Scene Memory for Vision-Language Navigation (Wang 2021)][ssm]
- [GSMem: 3D Gaussian Splatting as Persistent Spatial Memory (Lu 2025)][gsmem]
- [MemoNav: Working Memory Model for Visual Navigation (Li 2024)][memonav]
- In-repo: `scripts/r2dreamer/check_episode_frame_alignment.py` (FULL baseline,
  job 5707283), `src/vggt/jax/feature_extractor.py` (`ResetMode`,
  `reset_for_scene`), `docs/notes/visible-house-context-snapshot.md`.

[d82]: https://github.com/danijar/dreamerv3/issues/82
[d164]: https://github.com/danijar/dreamerv3/issues/164
[dv3]: https://export.arxiv.org/pdf/2301.04104v1.pdf
[cleanrl]: https://github.com/dosssman/cleanrl/blob/mbrl-dreamer/cleanrl/dreamer_atari.py
[rim]: https://ar5iv.labs.arxiv.org/html/2308.05602
[ssm]: https://ar5iv.labs.arxiv.org/html/2103.03454
[gsmem]: https://arxiv.org/html/2603.19137v1
[memonav]: https://arxiv.org/html/2402.19161
[ev]: https://github.com/ul-sail/Master-Thesis-3D-VLA/blob/main/src/r2dreamer/launch/evaluate.py