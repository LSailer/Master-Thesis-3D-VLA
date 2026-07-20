## Context

VGGT's DPT head emits, per frame, a dense point map `world_points (H×W×3)` with
per-pixel `confidence` (expp1, ≥1) plus RGB from the source image
(`src/vggt/jax/feature_extractor.py`, `VGGTExtractOutput`). The agent needs a
*persistent* per-house cloud, but VGGT's streaming aggregator KV cache is bounded
and evicts; `ResetMode.FULL` wipes state per episode while `ResetMode.PERSIST_SCENE`
resumes an attention stream per `scene_id` (with a noted unbounded camera-head cache,
HANDOFF.md §2/§4.2). The repo already has one external accumulator — a voxel
open-addressing **hash table** (`HouseContextPoseBuffer`, spatial hash with the
Teschner primes `73856093/19349663/83492791`, capacity + `overflow_count` +
`failed_insert_count`). Downstream encoders (MLP, `GnnHousePointsCameraEncoder`,
PointNet) consume its `(max_points, 6)` snapshot.

Three ways to accumulate over time therefore exist, not two:
1. **Internal to VGGT** — bounded rolling KV cache that *prunes/forgets*
   (repo `PERSIST_SCENE`; literature InfiniteVGGT).
2. **External hash table** — persistent, complete, but *leaks* (never reclaims a cell).
3. **External graph memory** — persistent, complete, *reclaims* via consolidation
   (this change).

Internal and external memory do *different jobs*: VGGT solves rolling consistency;
external memory solves persistence of what VGGT is designed to forget. Task 0
measures where the boundary is.

## Goals / Non-Goals

**Goals:**
- Measure how VGGT's point-cloud output accumulates/drifts at 3 scales (1 image /
  1 episode / multi-episode) and under FULL vs PERSIST_SCENE — and derive a
  criterion for *whether/where* external memory is needed (Task 0).
- Empirically decide accumulation precision (bf16 vs f32) rather than assume it
  (Task 1a).
- Compare hash table vs graph memory on a shared harness with a matched-budget
  protocol, producing a table + growth plots + frontier + recommendation (Task 1b).
- Isolate *mechanism* as the only variable between hash table and graph.

**Non-Goals:**
- No change to the training loop, agent behavior, or the encoder contract.
- Not replacing `HouseContextPoseBuffer`; it is the baseline.
- Not re-implementing InfiniteVGGT; it is literature positioning, not a deliverable.
- Not building loop-closure/pose-graph SLAM; consolidation is local
  connected-components, not global bundle adjustment.

## Decisions

### D0 — Task 0 gates the mechanism work
Characterization is the lead deliverable and answers the reviewer question "why not
just use VGGT's own rolling cache / InfiniteVGGT?" with measurement. If Task 0 shows
VGGT keeps the cloud consistent *within* an episode, hash-vs-graph is only run in the
multi-episode / past-eviction regime — which is exactly where the graph's
consolidation is its only plausible edge.

The three scales map onto VGGT's own memory architecture:

```
  1 image       single forward, no history      → raw point-map noise/confidence/yield
  1 episode     cache fills then EVICTS          → world-consistency across frames;
                                                   where does eviction drop geometry?
  multi-episode FULL wipe vs PERSIST_SCENE       → does persisting help or does
                                                   eviction/unbounded cam-cache break it?
```

### D1 — Both external mechanisms drink the same DPT stream
Hash table and graph memory both ingest `world_points + confidence + rgb` directly
from the DPT head (no input confound). The same **confidence gate** is applied to
both, so any FP difference is mechanism, not thresholding.

### D2 — Graph node = running centroid; edges enable consolidation
Chosen graph role: **edges for consolidation** (global merge the hash table cannot
do). Per-frame the graph does cheap local merge/spawn; at episode boundaries it
consolidates connected nodes that have *drifted* within the merge radius as
observations refined them — reclaiming capacity. The hash table's *mechanism of
loss* is that it never reclaims a cell; the graph's *mechanism of victory* is that
consolidation frees slots → higher coverage-per-slot at fixed capacity `N`.

Fixed-shape device state (JAX/jit discipline, mirroring `_VoxelContextState`):
```
  node_sum_xyz (N,3) f32   accumulate here (segment_sum) — NEVER bf16   [see D4]
  node_count   (N,)  i32
  node_rgb     (N,3) uint8  (like the hash table's store_rgb)
  occupied     (N,)  bool
  neigh_idx    (N,K) i32    fixed-degree k-NN edges = "the graph"
  lookup grid  (hash→slot)  optional spatial index for the merge query [see D3]
  published:   node_xyz = (node_sum_xyz / node_count).astype(bf16) at snapshot only
```

### D3 — Nearest-node lookup structure is a deliberate fork (settle in 1a)
"Find nearest node within r" can be done three ways, and the choice *is partly what
the speed experiment measures*:
- **(a) brute-force** dense P×N distances — the "pure" graph, no grid anywhere;
  genuinely different architecture, but O(N) per point ties ingestion speed to map
  size.
- **(b) hash-grid-accelerated** — fast, but makes the graph a *superset* of the hash
  table (it can't then win on raw speed and differs only by centroid + reclaim).
- **(c) KD-tree (jaxkd)** — rejected: the repo notes a CUDA k=1 segfault
  (`gnn_house.py`).

Leaning **(a) brute-force** for the cleanest scientific contrast ("grid quantization
vs no quantization anywhere"); 1a builds a minimal accumulator so this is decided
while the code is small.

### D4 — Precision: store bf16, accumulate f32 — but PROVE it (1a)
Prediction (not yet a result): bf16 accumulation risks (i) running-sum stall (large
accumulator + small increment rounds to no-op → centroid freezes) and (ii)
distance-test quantization (bf16 ulp ≈ 0.03 m at ~4–8 m, coarser than a cm-scale
merge radius → reintroduces the boundary artifact the graph exists to kill). Task 1a
tests the falsifiable hypothesis "bf16 accumulation degrades coverage/FP vs f32 on
this data." If bf16 is within ε → use bf16 (codebase default, simpler); if it
degrades → f32 earns its exemption *with evidence*. Store the published snapshot in
bf16 regardless (matches `store_xyz` and the encoder contract).

### D5 — Consolidation = bounded parallel connected-components
Episode-boundary (or occupancy-triggered) pass:
```
  1. mark edge (i,j) mergeable if ‖node_xyz[i]−node_xyz[j]‖ < r_merge  (over neigh_idx)
  2. parent[i] = min canonical id among mergeable neighbors ∪ {i}   (deterministic tie-break)
  3. repeat T rounds: parent = parent[parent]   (path-halving, bounded while_loop)
  4. segment_sum node stats into roots           (graph contraction)
  5. free non-root slots (occupied=False)         (RECLAIM capacity)
```
All fixed-shape and jit-safe. Cadence = **end-of-episode OR occupancy > threshold,
whichever first** — amortizes cost and fires exactly when revisits have accumulated;
the occupancy valve decouples cadence from episode length. Per-frame hot path touches
only local cells so it stays within a constant factor of the hash table; the extra
cost is amortized consolidation.

### D6 — Collapse the two knobs to one comparable knob
Tie `r_merge = r_assign = r`. Then `r` and the hash table's `voxel_size` are the
*same physical quantity* ("two observations within this distance are the same surface
point"), so both mechanisms have exactly one scale knob with identical meaning and
the sole remaining variable is the mechanism. This also kills chain-collapse (only
bites when `r_merge ≫ r_assign`). Consolidation still does non-trivial *temporal*
work at tied radii (nodes whose centroids drift together after refinement).
**Rigorous fallback** if the tie is challenged: sweep the 2D grid `(r_assign,
r_merge)` and compare Pareto *frontiers* — the knobs need not match; the frontier is
the comparable object.

### D7 — Metrics and the matched-budget protocol
- **Ingestion speed (pts/s):** jitted, warmed up, `block_until_ready`; report
  per-frame hot-path throughput **separately** from amortized consolidation cost
  (else cadence rigs the number).
- **False-positive rate:** additions on a frame whose correct answer is "add
  nothing" — ground truth via a **revisit oracle** (feed the same frame N times,
  or replay a ProcTHOR trajectory looping back to seen geometry); additions beyond
  the first pass are FPs.
- **Coverage / recall:** fraction of a **dense no-dedup oracle cloud**'s occupied
  space (fine reference voxelization) within ε of a stored point. No ground-truth
  mesh required.
- **Growth @ 3 scales:** cumulative stored points vs frames — within image, across
  episode, across episodes/scenes. Expected signature: hash table monotonic creep
  (leaks); graph **sawtooth** (creep within episode, drop at consolidation) below it.
- **Matched-budget comparison:** sweep each knob, plot coverage vs stored-point
  count (memory); read speed/FP *at matched budget*. Comparing a single (fine hash,
  coarse graph) pair is invalid.

## Risks / Trade-offs

- **Graph can't beat the hash table on raw ingestion speed** if hash-grid-accelerated
  (D3b): it's a superset. The bet is coverage-per-slot (reclaim) + lower FP, and the
  decision rule (speed-first) makes "hash table dominates" a very live, honest
  outcome — which must be reported plainly if it holds (proposal: "state if neither
  dominates").
- **bf16 self-sabotage** (D4): if accumulation precision is too low, the graph
  quietly reintroduces boundary artifacts and loses for the wrong reason. 1a exists
  to catch this before 1b.
- **Cadence-as-confound** (D5): per-frame consolidation artificially sinks the speed
  metric; the protocol reports hot-path and consolidation costs separately.
- **Two-knob explosion** (D6): mitigated by tying radii; frontier-over-grid fallback
  kept in reserve.
- **Determinism**: parallel merge needs a canonical tie-break or results depend on
  scatter order (non-reproducible).
- **Oracle fidelity**: coverage/FP depend on a trustworthy dense oracle and
  controlled revisit sequences; ProcTHOR gives repeatable trajectories.
- **Task 0 could moot 1b**: if VGGT's internal memory already suffices at all scales
  of interest, external mechanism choice matters little — which is itself a
  publishable finding, not a failure.
