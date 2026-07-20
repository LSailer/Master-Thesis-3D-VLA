## 1. Task 0 — Characterize VGGT output over time (gating; no new mechanism)

- [ ] 1.1 Stand up a read-only harness that streams frames through
  `src/vggt/jax/feature_extractor.py` and captures `world_points`, `confidence`, and
  RGB per frame, under both `ResetMode.FULL` and `ResetMode.PERSIST_SCENE`; include a
  flag-gated colored-PLY export path reusing the existing `save()` writer (off by
  default)
- [ ] 1.2 Single-image scale: record per-pixel point yield, confidence distribution,
  and point-map noise as the baseline
- [ ] 1.3 Within-episode scale: measure world-frame consistency of overlapping
  geometry across frames; identify the frame index where aggregator-cache eviction
  starts dropping previously observed geometry
- [ ] 1.4 Multi-episode scale: compare drift/consistency across episodes of one scene
  under FULL vs PERSIST_SCENE; note the unbounded camera-head cache behavior
  (HANDOFF.md §2/§4.2)
- [ ] 1.5 KV-cache saturation stress: stream many episodes of one scene under
  `ResetMode.PERSIST_SCENE` past the aggregator eviction budget; identify the
  saturation point and plot geometric degradation (drift, dropped regions, lost
  world-frame consistency) as a function of episode count
- [ ] 1.6 Visual export around saturation: with the diagnostic flag on, dump colored
  PLY snapshots as the stream approaches, reaches, and passes saturation, for
  inspection/comparison in a point-cloud viewer (CloudCompare/Blender/Open3D)
- [ ] 1.7 Write the regime criterion: state whether/where external persistent memory
  is needed; this decision gates Tasks 1a/1b (if internal memory suffices at all
  scales of interest, scope down or mark 1b moot)
- [ ] 1.8 Add literature positioning: internal rolling memory (InfiniteVGGT,
  arXiv:2601.02281) vs external persistent memory

## 2. Task 1a — Precision probe (gated on Task 0)

- [ ] 2.1 Build a minimal graph accumulator: running-centroid merge/spawn only, no
  consolidation, no edges (prototype under `src/prototyp/<feature>/` per convention)
- [ ] 2.2 Decide the nearest-node lookup structure: implement brute-force ("pure",
  leaning choice) and note the hash-grid-accelerated alternative; record the
  speed-vs-map-size implication
- [ ] 2.3 Run bf16-accumulate vs f32-accumulate on the identical stream; measure
  centroid drift, coverage recall, and false-positive additions
- [ ] 2.4 Decide the accumulation dtype from the data (bf16 if within epsilon, else
  f32 with evidence); publish snapshot in bf16 either way
- [ ] 2.5 Record the decision and carry the accumulator skeleton + harness forward
  to Task 1b

## 3. Task 1b — Hash table vs graph mechanism comparison (gated on Task 0 + 1a)

- [ ] 3.1 Complete the full graph memory: fixed-degree edges + episode-boundary/
  occupancy-triggered consolidation via bounded parallel connected-components with a
  deterministic canonical tie-break; capacity reclamation on merge
- [ ] 3.2 Tie `r_merge = r_assign = r` to a single scale knob matching the hash
  table's `voxel_size`
- [ ] 3.3 Build the shared benchmark: mechanism-agnostic ingestion with identical
  confidence gate; ingestion-speed timer (hot-path vs consolidation reported
  separately); revisit-oracle false-positive counter; dense-oracle coverage recall;
  growth curves at three scales
- [ ] 3.4 Add flag-gated diagnostic PLY export to the benchmark: per-episode and
  before+after each consolidation, matched checkpoints across mechanisms + the dense
  oracle, with optional attribute coloring (confidence / new-vs-existing / node id);
  verify the flag is OFF during timing runs so pts/s is not contaminated by disk I/O
- [ ] 3.5 Run the matched-budget protocol: sweep each mechanism's knob to trace the
  coverage-vs-memory frontier; read speed and false-positive figures at matched
  budget (keep the 2D `(r_assign, r_merge)` frontier-over-grid as rigorous fallback)
- [ ] 3.6 Restrict runs to the regime Task 0 proved external memory matters; use
  controlled revisit sequences (e.g., ProcTHOR) for the FP oracle
- [ ] 3.7 Produce the deliverable: comparison table + growth plots (3 scales) +
  coverage-vs-memory frontier + exported PLY sequences (CloudCompare C2C diffs of
  hash vs graph vs oracle); apply the decision rule (coverage and speed first, FP as
  tie-breaker); write the paper recommendation and state explicitly if neither
  mechanism dominates
