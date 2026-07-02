# Status: house-context work streams — current stand, open problems, next steps

*2026-07-02, compiled from three subagent scans: working-tree delta, artifact/docs
audit, correctness review.*

## Current stand

Five uncommitted work streams sit on top of `928bcb6` (33 dirty/untracked files):

| # | Stream | State |
|---|---|---|
| 1 | **Live buffer rewrite** — fixed-shape device voxel hash (`_VoxelContextState`, open addressing, jitted `add`), adapter rewired for live VGGT points, masked mean/max pooling in `HousePointsCameraEncoder`, `HOUSE_CONTEXT_MAX_POINTS` raised to **262 144**, new `HOUSE_CONTEXT_SIZE_KEY` | complete, tested, consistent |
| 2 | **Model-size presets** — `LATENT_PRESETS` 12m…400m table, parser default `12m` | complete, tested |
| 3 | **PointNet++ encoder scaffold** (`encoders/pointnet2.py`) | stub: every method `NotImplementedError`, unregistered, unreachable |
| 4 | **Graph-spectral experiments** (`src/prototyp/graph_house_context/`, helpers, 26 tests) | complete; GPU-validated (job 5696695); conclusions in [graph-spectral-house-context-experiment.md](graph-spectral-house-context-experiment.md) |
| 5 | Housekeeping — SKILL.md YAML fix, new `CLAUDE.md`, 2 stray `2026-07-02-*.txt` transcripts in repo root | trivial |

Experiment headline (GPU, H100): k-NN graph build **0.6 s** for 210k points; store
nodes / recompute edges wins storage; contour sampling beats stride on contour
fidelity by up to 34 %; block-GFT RGB coding modest (5× @ 22 dB); sparse GCN
inpaints masked RGB at **+17 dB, ~32 ms/step full-cloud**.

Test suite: **382 passed**, 45 skipped, **2 pre-existing failures** (both the
same stub-signature drift: `prepare_env_step`/`_fake_start_episode` missing the
new `packer`/`_packer` arg — `tests/r2dreamer/launch/test_evaluate_manifest.py`,
`tests/r2dreamer/test_trainer.py`; neither file touched by any stream).

## Open problems

**Fixed today (verified by tests, GPU re-run submitted as job 5705858):**
- ~~Self-loops from duplicate coordinates~~ — the bench cloud has **1.3 %
  exact-duplicate xyz** because `store_xyz` is bfloat16 (ulp > 1 cm beyond
  2.56 m), and jaxkd tie order made "drop column 0 = self" unsound. Fixed in
  `knn_graph.py` (swap-self-into-column-0) + regression test. GPU re-run (job
  5705858, COMPLETED) confirms all published numbers are stable within noise
  (exp2 chamfer identical to 4–5 decimals; exp4 21.67 dB unchanged) — the bug
  was structural, not numerical, at this duplicate rate.
- ~~exp3 export fractions outside `--keep-fractions` wrote all-black PLYs~~.
- ~~exp3 energy-mode tie handling overstated the oracle RD curve~~ (now
  `top_k`, bytes match kept rows).
- ~~exp4 NaN PSNR when zero nodes corrupted~~ (now fails fast).

**Open — correctness/design:**
1. **bfloat16 xyz store aliases voxel positions** (root cause of the
   duplicate issue, affects the *live* path too): dedup keys are exact int32,
   but the stored representatives collapse beyond 2.56 m, so the encoder sees
   position-aliased points. Options: float32 `store_xyz` (2× memory:
   50→100 MB at 2^23), or store exact int32 voxel keys + reconstruct centers.
   Needs a decision + micro-eval.
2. **Two pre-existing test failures** (packer stub drift) — mechanical fix.
3. **Graph API not yet padding-aware** (altitude finding): feeding the live
   fixed-shape snapshot (zero-padded) into `build_knn_graph` /
   `local_variation_scores` would let padding rows capture edges and scores.
   Live adoption needs a `valid: (N,) bool` mask threaded through, or
   size-sliced host-side use.
4. **`_house_context_snapshot` float32 stride math is safe only for
   capacity ≤ 2^24**, but `_validate_config` doesn't enforce it — add the
   guard.
5. **jaxkd CUDA extension segfaults on k=1 queries** (worked around: chamfer
   uses the pure-JAX path). Upstream issue; keep the workaround documented.
6. **Single-scene replay approximation** in `augment_replay_batch`
   (documented in the adapter docstring) — fine for L1, wrong for multi-scene.

**Open — cleanup (review findings, unverified but consistent across two
finder angles):** duplicated `REPO_ROOT`/`DEFAULT_PLY`/argparse blocks across
the four exp scripts; `degree_colors`/`score_colors` near-copies; ~7 copies of
the rgb01↔uint8 idiom; exp3 inlines PSNR while importing `rgb_psnr` unused;
`voxel_block_keys` duplicates `_quantize_points`; second ASCII PLY writer
duplicates the buffer's; exp2 rebuilds kd-trees it could share (~4× waste) and
rebuilds the graph exp1 saved; per-row Python PLY writing (~1–2 s per 210k
cloud); two stride-index implementations with no shared helper; `lexsort` in
`_unique_frame_voxels` could be one packed-key sort (~1 ms/step of the ~2 ms
budget). None block correctness; batch them as one cleanup pass.

**Open — process:** nothing is committed. Streams 1, 2, 4 are each coherent
and test-green; PointNet2 scaffold (3) is intentionally stubbed.

## Proposed way to continue

1. **Hygiene (½ day):** fix the two packer-stub test failures; gitignore the
   stray transcripts; commit in stream-sized commits (buffer+adapter+encoder;
   presets; graph experiments + docs; scaffold). Then run the cleanup batch
   (`/simplify`-style) over the experiment code in one commit.
2. **Decide the xyz-store dtype** (open problem 1): quick experiment — encode
   the same scene from float32 vs bfloat16 representatives, measure encoder
   input distortion + memory. If float32 wins, it also eliminates the
   duplicate-coordinate class of bugs everywhere downstream.
3. **Adopt contour-aware snapshots in the live path** (the experiment's
   practical win, now that the budget is 262k and pooling is masked): thread a
   validity mask through `build_knn_graph`/`local_variation_scores`, then add
   a config-flagged alternative to the even-stride snapshot. Scores can be
   refreshed lazily (every N adds) since the cloud changes slowly.
4. **Implement the PointNet2 seam as a sparse GNN** rather than a TF1
   PointNet++ port: exp4 proved segment_sum message passing at 262k
   nodes/6.7 M edges costs ~30 ms/step on H100, and jaxkd builds the graph in
   0.6 s/scene. `PointNet2FeatureEncoder`'s input/output contract
   ((max_points, 6) → (1024,)) can stay; fill it with 2–3 GCN/EdgeConv layers
   + masked pooling, benchmark against `HousePointsCameraEncoder` on an L1
   smoke run (`smoke_house_points_pose.sbatch` as template).
5. **Park block-GFT compression** — validated but not the storage bottleneck;
   keep as a thesis section (honest negative/modest result) rather than
   engineering effort.
