## Why

The agent needs a persistent per-house point cloud (RGB + XYZ), but VGGT emits a
*fresh point map per frame* and its streaming KV cache is **bounded** — it evicts
old geometry and, in the `PERSIST_SCENE` line of work, is explicitly designed to
*forget* ("catastrophic drift over long sequences" is what InfiniteVGGT,
arXiv:2601.02281, exists to fix via a bounded, pruning rolling memory). So before
choosing *how* to accumulate points externally, we do not actually know **how
VGGT's own output behaves over time** — within one image, across one episode,
across multiple episodes — nor **where VGGT's internal memory stops sufficing** and
an external persistent memory becomes necessary. Picking an external mechanism
(hash table vs graph) without that measurement is guessing.

## What Changes

- **Task 0 — Characterize VGGT output over time (gating).** A read-only
  measurement of how `VGGTExtractOutput.world_points` evolves at three scales
  (1 image / 1 episode / multi-episode), under both `ResetMode.FULL` and
  `ResetMode.PERSIST_SCENE`, to answer: *is external memory needed, and in which
  regime?* No new accumulation mechanism.
- **Task 1a — Precision probe (gated on 0).** A minimal graph accumulator
  (running-centroid merge only, no consolidation/edges) run bf16-accumulate vs
  f32-accumulate on the same stream, to *empirically decide* the accumulation
  dtype instead of assuming f32. Also settles the nearest-node lookup structure
  (brute-force "pure" vs hash-grid-accelerated).
- **Task 1b — Mechanism comparison (gated on 0 + 1a).** The full graph memory
  (radius-merge + edges + episode-boundary consolidation) vs the existing voxel
  hash table (`HouseContextPoseBuffer`), compared *only in the regime Task 0
  proves external memory matters*, on a shared measurement harness. Deliverable:
  comparison table + growth plots (3 scales) + coverage-vs-memory frontier + a
  paper recommendation, stating explicitly if neither mechanism dominates.
- **Decision rule (for 1b):** prioritize **coverage and ingestion speed** first;
  treat **false-positive rate** as a secondary tie-breaker.
- A new **graph scene-memory** mechanism is introduced as an artifact of 1a/1b,
  paralleling `HouseContextPoseBuffer`, not replacing it.

## Capabilities

### New Capabilities
- `vggt-output-characterization`: The measurement of how VGGT's per-frame point
  cloud accumulates/drifts over time at three scales and across reset modes, and
  the criterion for deciding whether/where external memory is needed.
- `scene-memory-benchmark`: The shared, mechanism-agnostic harness that ingests a
  VGGT point stream into a scene-memory backend and measures ingestion speed,
  false-positive additions (via a revisit oracle), coverage vs a dense oracle, and
  point-set growth at three scales — with a matched-budget (Pareto frontier)
  protocol so mechanisms are comparable.
- `graph-scene-memory`: A persistent external point-cloud memory whose nodes are
  running centroids of merged points and whose edges enable episode-boundary
  consolidation (drifted-close connected nodes merge, reclaiming capacity).

### Modified Capabilities
<!-- None. The voxel hash-table buffer (HouseContextPoseBuffer) is used as the
     baseline for comparison but its requirements are not changed by this
     investigation. -->

## Impact

- **New code (investigation-scoped):** measurement/benchmark harness; a minimal
  then full graph-memory accumulator paralleling
  `src/buffer/house_context_pose_buffer.py`. Prototyping follows
  `src/prototyp/<feature>/` per project convention.
- **Read/exercised, not modified:** `src/vggt/jax/feature_extractor.py`
  (`VGGTExtractOutput.world_points`/`confidence`, `ResetMode`,
  `_scene_cache_store`, aggregator eviction); `src/buffer/house_context_pose_buffer.py`
  (voxel hash-table baseline).
- **Literature positioning:** InfiniteVGGT (arXiv:2601.02281, internal *rolling*
  memory) vs external *persistent* memory; point-based fusion (Keller et al., 3DV
  2013) as the graph-memory pedigree; Teschner et al. (VMV 2003) as the origin of
  the hash-grid primes already in the code.
- **No training-loop or agent behavior change** is proposed here; this is a
  measurement + prototype investigation that produces a recommendation.
