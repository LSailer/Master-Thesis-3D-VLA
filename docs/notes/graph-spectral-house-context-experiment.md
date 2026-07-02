# Graph-Spectral House-Context Experiments

**Question.** Does representing the full house point cloud as a graph — nodes
= xyz voxel representatives, node attribute = RGB — help compactly store and
summarize the scene, per *Graph Spectral Image Processing* (Wiley/ISTE)?

- Ch 7 "Graph Spectral Point Cloud Processing" (Hu, Chen, Tian, pp. 229–267):
  graph construction (k-NN / ε-N, §7.2), spectral vs nodal filtering (§7.3),
  contour-aware downsampling with π_i ∝ ‖(L·X)_i‖ (§7.4.2.1, Chen et al. 2018).
- Ch 5 "Graph Spectral 3D Image Compression" (pp. 144–168): geometry coded by
  octree/tree, color coded by block-wise Graph Fourier Transform over a
  geometry-derived graph with Gaussian weights w_ij = exp(−d²/σ²) (§5.2–5.3).
  Laplacian eigendecomposition is O(N³) → block partitioning is mandatory at
  house scale (p. 167).

**Data.** Saved 1 cm buffer snapshot of an L1 scene:
`output/bench/house_context_50steps/bench_50steps_full_1cm/step_00000_context.ply`
(209 806 points, xyz + rgb). Code: helpers in `src/prototype_helpers/`
(`knn_graph.py`, `graph_ops.py`, `graph_metrics.py`, `graph_gcn.py`,
`ply_io.py`), scripts in `src/prototyp/graph_house_context/`, outputs in
`outputs/prototype/graph_house_context/`. Tests: `pytest tests/prototype_helpers -q`.

GPU is optional everywhere (`--cuda` uses the jaxkd CUDA kd-tree); Slurm
wrapper: `sbatch scripts/r2dreamer/slurm/graph_house_context.sbatch`.

---

## E1 — Graph construction (nodes = xyz, attribute = rgb)

```bash
JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp1_build_graph
```

Symmetrized k=16 NN graph via `jaxkd`, Gaussian weights, σ auto = mean
neighbor distance.

| quantity | value |
|---|---|
| nodes | 209 806 |
| directed edges (E = 2·N·k) | 6 713 792 |
| σ (auto) | 0.0131 m |
| node table (xyz bf16 + rgb u8) | 1.9 MB |
| edge table, COO int32+bf16 | 67.1 MB |
| edge table, implicit senders | 20.1 MB |
| build time | ~91 s CPU / **0.6–2 s H100** (`--cuda`, across 3 jobs) |

**Storage answer:** edges cost ~10× the node table even in the cheapest
layout. But k-NN edges are a *pure function of xyz* — they can be rebuilt at
load time instead of stored. Store nodes, recompute edges (0.6 s on GPU).

Artifacts: `exp1/stats.json`, `graph_k16.npz`, `degree_hist.png`,
`weight_hist.png`, `nodes_rgb.ply`, `nodes_degree.ply` (degree as viridis —
contours/edges of the house glow).

## E2 — Contour-aware downsampling vs even-stride (Ch 7 §7.4.2.1)

```bash
JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp2_contour_downsample
```

Scores π_i ∝ ‖(L·X)_i‖ (high-pass response of the coordinate signal), Gumbel
top-k sampling without replacement; baseline = the live pipeline's
`HouseContextPoseBuffer.resample_xyzrgb` at equal budgets. Two chamfer
metrics (the sample→full direction is 0 for subset sampling): overall
coverage = full→sample, contour fidelity = top-decile-score points→sample.

| budget | coverage contour [mm] | coverage stride [mm] | contour-fid contour [mm] | contour-fid stride [mm] |
|---|---|---|---|---|
| 4 096 | 24.6 | **23.5** | **22.1** | 23.2 |
| 16 384 | 13.9 | **13.0** | **11.3** | 12.7 |
| 65 536 | 6.6 | **5.8** | **3.9** | 5.9 |

**Reading:** the theory's trade-off, measured. Contour sampling gives up a
little uniform coverage (~5–13 % worse mean chamfer) and buys markedly
better structural fidelity (5 % → 34 % better on contour regions, growing
with budget). At the current 4 096-point encoder budget the effect is small;
it becomes decisive at 16k–64k budgets — relevant if
`HOUSE_CONTEXT_MAX_POINTS` is raised per
`docs/notes/house-context-full-buffer-options.md`.

**Note:** the first run of this experiment exposed an int32 overflow in
`HouseContextPoseBuffer.resample_xyzrgb` / `_house_context_snapshot`
(`arange(max_points) * point_count > 2**31` wraps negative and collapses
coverage). Fixed in `src/buffer/house_context_pose_buffer.py` with regression
tests; production would have hit it at >524 288 stored voxels with the 4 096
snapshot (full house ≈ 5 M voxels at 1 cm).

Artifacts: `exp2/metrics.{json,csv}`, `chamfer_vs_budget.png`, `scores.ply`
(log-score inferno colormap), `contour_*.ply` / `stride_*.ply`.

## E3 — Block-wise GFT compression of RGB (Ch 5 §5.2)

```bash
JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp3_gft_compress
```

0.5 m voxel blocks (buffer `_quantize_points` idiom), per-block dense k=8
graph → Laplacian → `eigh` (float32) → RGB spectral coefficients → keep a
fraction (`lowfreq` = honest codec, indices implicit; `energy` = oracle
bound) → PSNR + attribute-size estimate (bf16 coefficients) vs raw RGB
(3 B/point = 629 kB).

Measured (99 blocks at 0.5 m after 3 000-point chunking, median 2 438
points/block, 437 s CPU / 206 s H100, eigh-dominated; keep=1.0 roundtrips at
121.6 dB on CPU and 71.0 dB on GPU — float32 eigh accuracy is
backend-dependent, but both are visually lossless and validate the pipeline):

| mode | keep | PSNR [dB] | size [kB] | × vs raw RGB |
|---|---|---|---|---|
| lowfreq | 0.02 | 19.26 | 25.5 | 24.7 |
| lowfreq | 0.05 | 21.01 | 63.2 | 10.0 |
| lowfreq | 0.10 | 22.13 | 126.1 | 5.0 |
| lowfreq | 0.20 | 23.51 | 252.0 | 2.5 |
| lowfreq | 0.50 | 26.58 | 629.6 | 1.0 |
| energy | 0.10 | 24.26 | 168.1 | 3.7 |
| energy | 0.50 | 34.67 | 839.4 | 0.75 |

**Reading:** compression works but is modest — VGGT-reconstructed indoor
color carries texture and reconstruction noise, so it is far less
graph-smooth than the clean MPEG clouds the book benchmarks. More
importantly, raw RGB is only 630 kB of a 1.9 MB node table: attribute coding
is not the storage bottleneck at house scale. (No quantization/entropy stage
here — coefficients priced at raw bf16 — so these ratios are a lower bound.)

Artifacts: `exp3/metrics.json`, `rd_curve.png`, `block_hist.png`,
`recon_p0.05.ply`, `recon_p0.2.ply`.

## E4 — Sparse GCN RGB reconstruction (UvA JAX tutorial 7, sparse)

```bash
# CPU smoke:
JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp4_gcn_rgb_demo \
    --steps 100 --max-points 50000
# Full run (produced the numbers below), via sbatch scripts/r2dreamer/slurm/graph_house_context.sbatch:
python -m src.prototyp.graph_house_context.exp4_gcn_rgb_demo --cuda --steps 2000
```

Hand-rolled Flax GCN (weighted-mean `segment_sum` message passing over the
edge list — dense adjacency would be O(N²) ≈ 176 GB at this scale; no jraph
dependency). Task: mask 30 % of node RGB, reconstruct from neighbors.

Measured (full 209 806-point cloud, 2 000 steps, H100 `--cuda`; the CPU
smoke at 50k points/100 steps reached 19.7 dB):

| quantity | value |
|---|---|
| loss initial → final | 0.146 → 0.0204 (converged) |
| PSNR prediction on masked nodes | 21.67 dB |
| PSNR corrupted input on masked nodes | 4.55 dB |
| train time | 63 s (≈32 ms/step at 6.7 M edges) |

**Reading:** +17 dB over the corrupted input purely from neighborhood
message passing — the k-NN house graph carries enough spatial structure for
a GNN encoder to exploit, and a full-cloud training step costs only ~32 ms
on H100.

Artifacts: `exp4/metrics.json`, `loss_curve.png`, `corrupted.ply`, `recon.ply`.

---

## Conclusions

1. **Store nodes, recompute edges.** The buffer's node table (xyz bf16 + rgb
   u8, 9 B/point ≈ 1.9 MB for this scene, ≈ 45 MB for a full 5 M-voxel house)
   is already the compact representation; a k=16 edge table costs 10× more
   and is a pure function of xyz, so it should be rebuilt at load/encode time
   (0.6 s with `--cuda` jaxkd on H100, ~91 s CPU), not persisted.
2. **Contour-aware downsampling is the graph method with practical value**:
   it preserves structural detail markedly better than the current stride
   snapshot at 16k+ budgets, at a small uniform-coverage cost. Candidate for
   the snapshot path if the encoder budget grows.
3. **Block-GFT RGB coding validates but does not pay at house scale**: 5×
   attribute compression at 22 dB; raw RGB is only a third of the node
   table. The eigendecomposition cost (206 s on H100 for 210k points)
   confirms the book's O(N³) warning — blocks are mandatory, and even then
   it is the dominant cost.
4. **The graph is learnable**: +17 dB RGB inpainting on masked nodes (2 000
   GCN steps, full cloud, ~32 ms/step on H100) supports a future
   sparse-GNN house encoder (the locality
   upgrade `src/r2dreamer/encoders/pointnet2.py` anticipates), using exactly
   this segment_sum message-passing pattern.
5. **Side discovery:** the experiment exposed and fixed an int32 stride
   overflow in `HouseContextPoseBuffer` that would have silently corrupted
   production snapshots beyond ~524k stored voxels.
