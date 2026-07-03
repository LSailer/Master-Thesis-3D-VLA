# Experiments: VGGT cache persistence across episodes and agents

> Companion to the canonical handoff at
> `docs/notes/cache-persistence/handoff.md` (with source-file reference
> index at `docs/notes/cache-persistence/references.md`; the stub
> `HANDOFF.md` here just redirects). Each experiment states the
> question, the prediction, the harness, the metric, and the go/no-go
> criterion. Build all harnesses by **extending**
> `src/prototype_helpers/vggt_cache_diagnostics.py` and driving from
> `src/prototyp/prototype_vggt_cache_diagnostic.py` — do not fork a parallel
> harness. Run GPU work via SLURM (`sbatch`/`srun`); the login node is
> CPU-only and `JAX_PLATFORMS=cpu` is the local fallback (see memory
> gpu-via-slurm).

## Conventions

- **Cache state** = the four extractor fields `_past_kvs_padded`,
  `_last_scores`, `_past_kvs_camera`, `_frame_idx`, read with
  `read_cache_state` (already returns aggregator/camera `valid_len` +
  `frame_idx`).
- **Baseline (fresh-reset)** = current behaviour: `extractor.reset()` at
  every episode boundary.
- **Treatment (persisted)** = `save_cache`/`load_cache` per scene, no global
  reset (HANDOFF §4.3 — implement the API first, even as a no-op behind a
  flag, so these experiments have something to call).
- **Drift metrics** (all already implemented in `vggt_cache_diagnostics.py`):
  `point_delta_stats` (mean, p50/p90/p95/p99/max + threshold counts at
  1e-4…1e-1 m), `camera_l2`, `camera_max_abs`, `global_token_cosine`. Use
  these verbatim so results are comparable to the existing 20-step plots.
- Log to `outputs/prototype_cache_persistence/<EXP>/` (ignored output dir;
  do **not** write into `src/prototyp/`, per its `AGENTS.md`).

---

## E0 — Pre-flight: confirm the camera-head sliding window is bounded

**Question:** With the sliding-window eviction landed (HANDOFF §4.2b,
2026-07-02), does "no reset" now run past `max_camera_frames` with the
camera cache staying bounded at `_CAM_MAX`, instead of crashing?

**Why first:** The former `RuntimeError` blocker is resolved, but the
eviction is host-orchestrated (one concat per trunk block per frame once
full) — confirm it actually fires at frame 1024, holds `valid_len` flat at
`_CAM_MAX`, and measure the wall-clock cost before scaling to 1 M steps.

**Harness:** Single episode, never reset, step well past
`max_camera_frames` (e.g. 1500 frames). Record `_frame_idx`, camera
`valid_len`, and per-step wall-clock from frame 1000 → 1500.

**Prediction:** at frame 1024 the evictor fires; camera `valid_len` stays
flat at `_CAM_MAX` for all frames ≥ 1024; no crash; per-step time may rise
slightly (R5 concat cost).

**Go/no-go:** If `valid_len` stays bounded and no crash → eviction works,
proceed to E1. If per-step time regresses materially vs. the 158 ms/step
baseline → profile the concat cost and consider the circular-buffer
variant (HANDOFF §4.2b note) before E2b.

---

## E1 — Per-scene persistence across episodes (the core ask)

**Question:** When episodes of the **same scene** resume the VGGT cache
instead of resetting, is the per-frame VGGT output better (lower drift to a
self-consistent reference, more complete map) than fresh-reset, and does
the cache stay bounded?

**Harness:** One scene, run **K episodes** (e.g. K = 5), each ≤ 500 steps.

- Arm A (baseline): `reset()` at each episode start.
- Arm B (treatment): `load_cache(scene_id)` at each episode start
  (fresh on episode 0); never global-reset.
- At every step: record `read_cache_state` (agg/cam `valid_len`,
  `frame_idx`) + the four drift metrics vs. that episode's frame-0 output
  (within-episode drift) and vs. episode-0 frame-0 (cross-episode drift).

**Prediction:**
- Arm B aggregator `valid_len` grows to the per-block budget (≈ 8333 at the
  training `VGGT_STATIC_BUDGETS`, or ≈ 50 000 at the extractor default) and
  **plateaus** — eviction holds it. Camera `valid_len` grows linearly
  (unless E0 forced eviction) → must plateau too or E1 is invalid.
- Cross-episode drift (Arm B vs. Arm A): Arm B should show **lower**
  camera-pose spike on the first few frames of episode k>0 (no re-anchoring
  jump) and a more coherent point map. But R2 (teleport) may show up as a
  transient tear on frames 1–3 of the resumed episode.

**Metrics to report:**
1. Aggregator/camera `valid_len` vs. step, both arms, all episodes
   (proves boundedness).
2. First-frame-of-episode camera L2 (Arm B should not have the reset
   spike Arm A has).
3. Map coherence: feed both arms' `world_points` into
   `HouseContextPoseBuffer` and report unique-voxel count + cross-episode
   IoU@10 cm (reuse `scripts/r2dreamer/check_episode_frame_alignment.py`).
   Arm B should approach the alignment check's ALIGNED criterion; Arm A is
   the known ~0.07–0.14 IoU baseline.

**Go/no-go:** Arm B must (a) stay bounded, (b) not crash, (c) beat Arm A on
cross-episode IoU. If it does → per-scene persistence is viable for level 1.
If cross-episode IoU is still ~0.1 → the resume teleport (R2) dominates and
**persistence alone is not enough**; pair it with the Option-C reanchoring
(HANDOFF §0 cross-ref) rather than replacing it.

---

## E2 — 1 M steps with the same cache (the long-horizon stress)

**Question:** What happens to VGGT output quality and cache state over
~1 M consecutive frames with **no reset at all** (the level-1 "no reset"
regime, single continuous stream)? Does attention drift, collapse, or stay
bounded-and-stable?

> This is only meaningful **after** E0's camera-head eviction is in, else it
> crashes at 1024. Use a synthetic continuous walk (not random episode
> teleports) so the temporal-assumption violation is minimised — E2 isolates
> *long-horizon drift*, not *teleport* (that is E1/R2).

**Harness:** One long Habitat episode is not 1 M steps (max ~500). Two
routes:

- **E2a (controlled, cheap):** replay a looping camera path (or the same
  scene walked repeatedly without `reset`) for N = 100 k frames on GPU,
  logging every 1 k-th frame. This is the direct extension of the existing
  `run_cached_sequence` (currently 20 steps) to 10⁵ steps.
- **E2b (training-scale, expensive):** a level-1 training run with the
  no-reset flag, 1 M env steps, ~44 h on one H100 (memory: step cost
  ≈ 158 ms/step). Log cache state + a periodic held-out frame's drift every
  5 k steps. Use the `afterany` chain to 2 M if 1 M is incomplete (memory
  house-points-pose-step-cost).

**Prediction:**
- Aggregator `valid_len` plateaus at budget within the first ~budget/P
  frames (≈ 6 frames at 8333/block) and stays flat for the full 1 M —
  eviction is stable.
- Point drift (adjacent, vs. frame-0): the *from-frame-0* drift will grow
  (the anchor frame is 1 M frames old and increasingly irrelevant), while
  *adjacent* drift stays small. This separates "the cache is unstable"
  (bad) from "the cache just forgets the far past" (fine, expected).
- Global-token cosine vs. frame-0: decays; vs. adjacent: stays near 1.0.
- Risk: the diversity-eviction feedback (`_calculate_dynamic_budgets`) may
  collapse a block's budget toward its floor (`P+1`) if `last_scores` saturate
  high, starving that block. Watch for any block whose `valid_len` hits
  `P+1` and stays — that is collapse.

**Metrics to report:** `valid_len` per block over time (all 24), adjacent
vs. frame-0 point/camera/token drift curves, min per-block `valid_len`
over time (collapse detector), wall-clock per step (regression vs. the
158 ms/step baseline — eviction cost may rise).

**Go/no-go:** Adjacent drift stays bounded AND no block collapses for the
full run → long-horizon no-reset is safe for level 1. If a block collapses
or adjacent drift diverges → the cache needs a refresh/anchor-update
mechanism (R3) before training with it.

---

## E3 — Same sofa, different agents (cross-agent cache retrieval)

**Question:** If agent A builds a VGGT cache in scene S and agent B starts a
new episode in scene S, can B **load A's saved cache** and still produce
good geometry of the sofa — i.e. does the cache "remember" the sofa across
agents, or does the anchor-frame/pose mismatch (R2/R4) break it?

**Harness:** Two extractor instances, one scene.

1. Agent A: run one episode in scene S, `save_cache()` → snapshot SA.
2. Agent B (fresh extractor): `load_cache(SA)` at episode start, run a
   short episode in scene S from a **different start pose**. Compare three
   arms:
   - B-fresh: B.reset() (baseline, ignores A's cache).
   - B-load: B.load_cache(SA) (treatment).
   - B-reobserve: B.reset() then walk to the sofa the same way A did
     (control: how good is "seeing the sofa again from scratch"?).
3. On the frames where B sees the sofa, measure sofa-region point drift
   vs. A's sofa frames, and whether the sofa's tokens are in B's loaded
   cache (anchor or retained candidate) vs. evicted.

**Prediction:**
- If the sofa was in A's **anchor frame**: its tokens are permanent and
  load into B → B's first sofa frames should show low drift to A's sofa.
  But B's start pose differs, so B's *current* frame is mis-registered
  against A's anchor frame (R4) → the attention may not actually use those
  anchors coherently. Expect: sofa tokens are *present* but *misaligned*.
- If the sofa was **not** in A's anchor frame: the sofa's tokens are
  candidates; whether they survived eviction depends on whether they were
  diverse enough. Re-observation by B adds redundant sofa tokens that
  **eviction discards** (HANDOFF §2) → the cache keeps A's representative,
  does not improve it.

**Metrics to report:** sofa-region point delta (B-load vs. A, and B-fresh
vs. A), presence of A's sofa tokens in B's loaded cache (count surviving
anchors + candidates), and the alignment of B's frame-0 to A's anchors
(camera L2 on frame-0).

**Go/no-go:** B-load beats B-fresh on sofa geometry AND the loaded sofa
tokens are actually used by attention → cross-agent sharing is worth a
shared per-scene cache store. If B-load is no better than B-fresh (or
worse, due to anchor mismatch) → **do not share across agents**; keep
persistence per-scene/per-agent only. Either outcome answers the user's
question "does the cache remember it" concretely.

---

## Execution order and dependencies

```
E0 ──► (camera eviction if needed) ──► E1 ──► E2b (training-scale)
                                     │
                                     └──► E2a (controlled) ──► E3
```

- E0 is gating: nothing long runs until the camera head is bounded.
- E1 is the core decision: per-scene persistence yes/no.
- E2a is the cheap long-horizon probe; E2b is the expensive confirmation.
- E3 is independent of E1/E2's outcome for its harness, but its
  *interpretation* depends on E1 (if per-scene persistence fails, cross-agent
  is moot).

## Shared harness work (do once, up front)

1. Add `save_cache`/`load_cache`/`reset_for_scene` to
   `feature_extractor.py` (HANDOFF §4.3) — all experiments need it.
2. Extend `VGGTCacheDiagnostic` with: multi-episode driver, cross-extractor
   cache transfer, per-block `valid_len` logging, a collapse detector
   (min block `valid_len`), and a sofa-region masked point-delta. Land these
   in `src/prototype_helpers/vggt_cache_diagnostics.py`.
3. Add a camera-head eviction path if E0 demands it (HANDOFF §4.2b).
4. New prototype drivers under `src/prototyp/cache_persistence/`
   (`e1_per_scene.py`, `e2_long_horizon.py`, `e3_cross_agent.py`) — these
   are experiment drivers, not reusable helpers, so they belong here per
   `src/prototyp/AGENTS.md`.

## Reporting

Each experiment writes a short result note to
`docs/notes/cache-persistence-<EXP>.md` with the predicted-vs-actual table
and the go/no-go verdict, linked back to this file. Do not inline results
in this document — keep it the plan.