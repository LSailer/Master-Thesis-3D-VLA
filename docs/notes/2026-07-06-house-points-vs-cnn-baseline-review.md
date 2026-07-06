# Deep review: VGGT house-points-pose (live buffer) vs. CNN image baseline

**Date:** 2026-07-06 · **Run under review:** job 5741728
(`vggt-house-points-pose-l1-live`, `run-5741728`, ~646k/2M steps at time of
writing; sibling seed 5764453 same config) · **Baseline:** image-only CNN L1
(`output/r2dreamer-curriculum-l1/run-4194043` + seed `run-4367942`, 2.4M steps,
completed).

## 1. What is actually being compared

The two variants do **not** share any input modality — this is the single most
important caveat for interpreting the numbers:

| | CNN baseline | `vggt_house_points_pose` (5741728) |
|---|---|---|
| RGB image | ✅ 64×64, only input | ❌ **not an encoder input at all** |
| Camera pose (9,) | ❌ | ✅ MLP branch |
| House point map | ❌ | ✅ (262144, 6) snapshot + valid count |
| Encoder | `ConvEncoder` (`src/r2dreamer/encoders/cnn.py:28`) | `HousePointsCameraEncoder` (`src/r2dreamer/encoders/mlp.py:105`) |
| Embedding | 1024 (flattened 4×4×64 conv map, no projection) | 2048 (concat camera 1024 ⊕ house 1024, no gate) |
| Encoder params | ~0.1M conv stack | ~1.65M (camera branch 1.06M, point head 0.59M) |

In the house-points run the RGB frame is used **only upstream** — VGGT lifts it
to world points that feed the per-scene buffer. The world model itself is
image-blind and must navigate from ego-pose + a mostly-static map. So the run
answers "how far does *map + pose alone* get you", not "does the map *add* to
vision".

## 2. Results at matched step windows (training episodes, L1 chair-only)

Window = env steps 400k–650k (the house run's most recent quarter):

| Run | success | SPL | n episodes |
|---|---|---|---|
| CNN `run-4194043` | **0.696** | **0.498** | 930 |
| CNN `run-4367942` | **0.717** | **0.520** | 969 |
| house-points `run-5741728` | 0.338 | 0.237 | 621 |

CNN baseline end-of-training (2.15–2.4M): success 0.671, SPL 0.474 — it
plateaus by ~500k steps.

House-points trend (thirds of the run so far): success 0.144 → 0.210 → 0.351,
`metrics/sr_mean` 0.090 → 0.156 → 0.208 (last 0.235), `dtg_mean` 2.14 → 1.86.
**Still climbing at 646k steps** where the CNN had already plateaued, but at
roughly **half** the baseline's success and SPL. Extrapolation to 2M is
uncertain; crossing the CNN plateau would require the current slope to hold
for another ~500k steps, which the flattening sr_mean curve does not suggest.

Two secondary observations:

- `loss/dyn` = `loss/rep` **rises** through the house run (4.2 → 5.8, last
  6.9) while the CNN baseline's is flat (~2.86 for 2.4M steps). The world
  model finds the (pose, map-snapshot) observation stream progressively harder
  to predict — consistent with a nonstationary global snapshot being broadcast
  into every replay batch (`hybrid_adapter.py:767`, see §4).
- Throughput ~171 ms/step (5.9 fps) — in line with the known ~219 ms
  production-shape cost, dominated by VGGT inference that the baseline
  (~28 ms/step) doesn't pay.
- Older baseline `run-4119014` (actfix rerun, Apr 27) only reached 13%
  success; the May runs `4194043`/`4367942` at 66–72% supersede it. Use the
  May pair as the canonical L1 CNN reference.

## 3. The "simple flatten" question

Current aggregation (`mlp.py:136-177`) is already the near-simplest
permutation-invariant design: shared point-MLP (2 × Dense(256)+RMSNorm+SiLU)
→ masked mean ⊕ max pool → Dense(1024). PointNet-lite, 0.59M params.

A flatten head would replace pooling with `reshape(-1) → Dense`. Measured
feasibility and stability (experiment below):

| Head | Params | Input coverage |
|---|---|---|
| point-MLP + pool (prod) | 0.59M | all 262144 snapshot points |
| flatten 262144 pts → 1024 | **1.61B — infeasible** | all |
| flatten 4096 pts → 1024 | 25.2M (42× prod) | 1.6% of snapshot |

**Stability experiment** (`flatten_vs_pool_stability.py`, CPU/JAX): a 5.4M-point
synthetic house cloud ingested in exploration order, snapshotted at growing
fill levels with the production even-stride rule
(`house_context_pose_buffer.py:254-283`), embedded with fixed random-weight
versions of each head. Cosine similarity of each snapshot's embedding to the
final full-map embedding:

| buffer size | pool_mlp | flat_raw | flat_sorted |
|---|---|---|---|
| 100k | 0.957 | 0.728 | 0.803 |
| 262k | 0.965 | 0.728 | 0.838 |
| 500k | 0.978 | 0.691 | 0.859 |
| 1M | 0.934 | 0.562 | 0.845 |
| 2M | 0.971 | 0.664 | 0.815 |

While the buffer grows, the even-stride snapshot keeps reassigning which point
lands in which slot; a flatten head is slot-position-sensitive, so its input
churns (cos ≈ 0.56–0.73 to the converged map, and 0.66–0.89 between
*consecutive* snapshots). Pooling is permutation-invariant, so the embedding is
stable (≥ 0.93) even when only 2% of the final map has been seen. Sorting
points by voxel key before flattening recovers some stability (≈ 0.80–0.86)
but stays well below pooling, at 42× the parameters.

**Verdict: keep the pooled head; don't spend a run on naive flatten.** The
one *principled* flatten variant is what the CNN baseline itself does —
flatten a **fixed spatial grid**. The equivalent for house points is
rasterizing the buffer into a BEV/voxel grid and flattening a conv map over
it (the `wp_dense_cnn` direction), which preserves slot-position meaning by
construction. That is the flatten experiment worth running, not
`reshape(points)`.

## 4. Buffer mechanics findings (affect any aggregation choice)

- **Growth does not saturate in the smoke horizon.** All four smokes agree:
  ~1.73M voxels @ 512 steps → ~4.1M @ ~1–2k steps (fill 0.49 of the 2^23
  capacity), still climbing ~2k voxels/step. At that rate the buffer hits
  capacity within the first few thousand steps of the 2M-step run. With **no
  eviction** (`_insert_unique_voxels` drops new voxels once full,
  `house_context_pose_buffer.py:127-219`), the "live" map effectively
  freezes during prefill — everything the agent sees afterwards never enters
  the map. For a static house this is survivable, but it means (a) early VGGT
  noise/misalignment is locked in forever, and (b) the `overflow_count`
  diagnostic is the number to check in the final manifest.
- **1 cm voxels are too fine for the budget.** 8.4M × 1 cm³ voxels is mostly
  surface micro-detail; the snapshot then even-strides 4.1M+ stored points
  down to 262k (≥ 16:1 discard). A 2–4 cm voxel would let the *whole* house fit
  both capacity and snapshot, making the observation truly stable and the
  even-stride subsample lossless.
- **The go/no-go metric is invisible mid-run.** `house_buffer/points_growth`
  is written only by `_log_adapter_summary` at end of run
  (`trainer.py:795-812`). For a 2-day job the growth curve — explicitly named
  in the config as the go/no-go — can't be checked until the job finishes.
  Logging `adapter.diagnostics()` every N steps alongside train metrics would
  fix this cheaply.
- **One global snapshot is broadcast across the replay batch**
  (`hybrid_adapter.py:767-780`): every sequence in a batch, whatever its age,
  is paired with *today's* map. Early-run transitions were collected under a
  sparser map than the one they're trained with — a moving-target effect that
  plausibly contributes to the rising `loss/dyn`.

## 5. Recommendations

1. **Keep** point-MLP + masked pooling; reject naive flatten (params +
   stability, §3).
2. **Let 5741728/5764453 finish** — the run is still improving and is the
   cleanest measurement of map+pose-only navigation; judge by final manifest
   (`overflow_count`, growth curve) not exit code.
3. **Next variant should be additive, not substitutive:** image CNN branch ⊕
   house-points branch (gated, as in `HybridEncoder`), so the map is tested as
   *extra* information against the strong 0.70-success CNN baseline rather
   than as a replacement for vision.
4. **Coarsen voxels to 2–4 cm** (or add eviction) so the map fits capacity and
   the snapshot stops discarding ≥ 94% of stored points.
5. **Log buffer diagnostics mid-run** (periodic `diagnostics()` rows in
   `metrics.csv`).
6. If a flatten-style head is still wanted, do it as **BEV rasterization +
   small CNN** (fixed grid → flatten is then well-posed), reusing the
   `ConvEncoder(input_kind="world_points")` machinery.

## Appendix: provenance

- Metrics windows computed from long-format `metrics.csv` (step, metric,
  value); files are not step-sorted — sorted before aggregation.
- Stability experiment script:
  `docs/notes/2026-07-06-house-points-vs-cnn-baseline-review-assets/flatten_vs_pool_stability.py`
  (synthetic house cloud: floor/walls/12
  furniture boxes, 1 cm voxel dedup, angular exploration order; fixed
  random-weight heads, bfloat16 compute; cosine in float32).
- Smoke growth curves: `output/runs/r2dreamer-curriculum-l1-vggt-house-points-pose-live/smoke/slurm-{5738008,5738777,5740085,5764452}.out`.
