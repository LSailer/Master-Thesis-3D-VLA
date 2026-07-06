# GNN house-context encoder — results & decisions (distilled 2026-07-03)

Distilled from `src/prototyp/gnn_house_encoder/` (RESULTS.md, DECISION_LOG.md)
per the prototype lifecycle. Code graduated to
`src/r2dreamer/encoders/gnn_house.py`; tests in
`tests/r2dreamer/test_gnn_house_encoder.py`.

## What it is

The house branch of `HousePointsCameraEncoder` replaced by a graph branch:
even-stride subsample of the live house snapshot to 4096 nodes → brute-force
k-NN (k=8, dense `(M,M)` + `top_k`, jit/grad-safe, avoids the jaxkd CUDA
segfault) → Gaussian edge weights → 2 GraphSAGE-style weighted-mean layers
(`segment_sum`, Dense+RMSNorm+silu, float32) → mean‖max pool → Dense(1024).
Output contract unchanged: `(B, 2048)` float32 (camera 1024 ‖ house 1024)
into the RSSM posterior.

Variant `GnnEdgeHousePointsCameraEncoder` (encoder_type
`gnn_edge_house_points_pose`): per-edge Dense over `[x_j, x_j − x_i,
p_j − p_i]` before the weighted mean (relative-position messages) plus
residual adds on width-preserving layers.

## Numbers worth citing

| Evidence | Result |
|---|---|
| 50k paired run, GNN (job 5736907) vs MLP (5736908), seed 42, H100 | Both completed, zero NaN/inf; softspl 0.099 vs 0.102 @50k (parity); SR = 0 both (too early); GNN transiently ahead @25k (0.122 vs 0.084) |
| Step-time overhead of graph branch (600+600 smoke, 5736062) | +10% vs MLP house branch (173 vs 158 ms/step) |
| EdgeConv variant overhead (canonical smokes 5744355/5744356) | +1.3% GPU step time vs baseline GNN (142.3 vs 140.5 ms/step) — the 65k-edge Dense is essentially free on H100 (CPU showed +70%, a misleading proxy) |
| Canonical smoke wall time | 1000 prefill + 2000 train = 12:08; configs now use 4500 train ≈ 20 min as the stability gate |

## Design decisions (with literature)

- **Relative-position edge messages** are the highest-value upgrade per the
  literature (PosPool arXiv:2007.01294: essentially all gain comes from
  encoding `p_j − p_i`; EdgeConv arXiv:1801.07829); attention/GAT is not the
  active ingredient. Implemented as the `edgeconv` variant.
- **Depth stays 2 layers** — GCN-family models peak at 2-3 layers, degrade
  past ~4 from oversmoothing (arXiv:2212.10701; PairNorm arXiv:1909.12223);
  residuals mitigate (arXiv:2501.00762), hence `residual=True` in the variant.
- **mean‖max readout kept** — learned readouts have contested evidence
  (pro: arXiv:2211.04952; con: arXiv:2406.09031); Set2Set rejected (LSTM
  iterations, jit-hostile).
- **Even-stride sampling kept** — random/stride ≈ FPS at a 4096/262k budget
  and FPS is a sequential-argmax latency bottleneck under jit
  (RandLA-Net arXiv:1911.11236; ISPRS 2025 sampling evaluation).
- **Default stays `sage`** — it is the 50k-validated config; `edgeconv` is
  registered (`habitat-l1-gnn-edge-house-points-pose`) for the next 50k
  comparison round rather than replacing a validated default on toy evidence.

## Operational learnings

- Judge habitat runs by `MANIFEST.json` status + checkpoints, never SLURM
  exit codes: GL teardown SIGABRT poisons exit codes (`FAILED 134`) after
  successful completion. Canonical configs guard via
  `R2DREAMER_HARD_EXIT_ON_FINISH=1` (`_base.yaml`); legacy sbatch files don't.
- Long-format `metrics.csv` is not step-sorted — use max(step), not last row.
- Launch smokes via `scripts/slurm/launch.sh
  gnn{,_edge}_house_points_pose_l1_live --smoke`.

## Open questions

- SR separation needs a longer horizon (500k+ steps); softspl parity at 50k
  says the GNN carries at least as much usable signal at +10% cost.
- `house_buffer/points_growth` reached 3.88M points by step ~2050 in both
  smoke arms — the PERSIST_SCENE saturation question is tracked separately
  (see docs/notes/visible-house-context-snapshot.md), not GNN-specific.
- Attention readout remains an untried, literature-plausible ablation.
