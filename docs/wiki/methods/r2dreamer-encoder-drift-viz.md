# R2Dreamer encoder-drift visualization (L1 VGGT, ckpt 300k)

**Status**: baseline analysis complete. Interactive 4-panel notebook deferred — see [Status & next steps](#status--next-steps).

## Goal

Make encoder drift, geometry loss, and representation collapse visible in R2Dreamer's world-model latent by comparing it against the VGGT 3D outputs that feed into it, frame-by-frame, on a matched success / near-miss episode pair from the same eval run.

The thesis question this addresses: **does R2Dreamer's RSSM preserve the geometric structure VGGT provides, and if not, where does it break?** Open issues this speaks to: [#70](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/70), [#87](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/87), [#89](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/89), [#90](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/90), [#59](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/59), [#64](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/64).

## Setup

| Item | Value |
|---|---|
| Run | `output/runs/r2dreamer-curriculum-l1-vggt/baseline-actent3e-4/` |
| Checkpoint | `step_000300000.pkl` |
| Curriculum | L1 (`fK2vEV32Lag` / chair only); 7499 train episodes / 834 eval, intersection = 0 |
| Pair-pick run | `debug/eval-pick-20ep/` (20 eps, SR = 5/20) |
| Instrumented re-roll | `debug/viz-pair-a/` (12 eps, SR = 5/12, 1.64 GB dumped) |
| Selected pair | **ep7** (success, 135 steps, SPL 0.96) + **ep1** (near-miss, 500 steps, ends 0.94 m from goal — **4.7× outside** the 0.2 m success radius; a final-approach failure) |
| Selected pair | **ep7** (success, 135 steps, SPL 0.96) + **ep1** (near-miss, 500 steps, ends 0.94 m from goal — *inside* the 1.0 m success radius without firing STOP) |

Pair-pick must come from the same run as the dumps because eval is non-deterministic on H100 ([#101](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/101)).

## Three projections

R2Dreamer has no 3D-decoder head — only scalar heads (reward, continue, actor, critic). "What the model has encoded" cannot be rendered directly; it must be probed or compared via similarity.

### A) Linear ridge probes — `latent → world_points`

Three probes via Ridge regression with α-sweep (α ∈ {1, 10, 100, 1000, 10000}) on standardized features:

- **probe_feat**: `feat (2560)` → `world_points_flat (4107)`
- **probe_deter**: `deter (2048)` → `world_points_flat (4107)`
- **probe_stoch**: `stoch (512)` → `world_points_flat (4107)`

Trained on 10 episodes (ep0, 2, 3, 4, 5, 6, 8, 9, 10, 11 = 3361 steps). Held out: ep7 + ep1.

Implementation: `scripts/debug_viz/fit_probes.py` (commit `1dccac7`).

### B) Spatial probe-error timeline — RMSE per step

For each held-out episode, plot per-step RMSE in metres for all three probes over the trajectory. Uses the predictions saved by `fit_probes.py`. Reveals whether reconstruction error is **localized** (one bad stretch = transient drift) or **monotonic** (compounding drift over the rollout).

### C) Temporal similarity matrices — T×T cosine

For each held-out episode:

- `S_VGGT[i,j] = cosine(world_points_i_flat, world_points_j_flat)` — VGGT's view of "frame i looks like frame j"
- `S_DREAMER_feat[i,j] = cosine(feat_i, feat_j)`
- `S_DREAMER_deter[i,j] = cosine(deter_i, deter_j)`
- `diff_feat = S_VGGT − S_DREAMER_feat`
- `diff_deter = S_VGGT − S_DREAMER_deter`

Plus quantitative summary: Pearson and Spearman correlation of off-diagonal entries (Dreamer-vs-VGGT agreement on relative similarity).

Implementation: `scripts/debug_viz/build_similarity.py` (commit `d64a647`).

## Findings — three coherent drift signals

| Metric | ep7 (success, 135 steps) | ep1 (near-miss, 500 steps) | Interpretation |
|---|---|---|---|
| **probe_feat RMSE** (m) | 0.228 | 0.367 | Encoder decodes geometry less faithfully during failure |
| **probe_feat R²** (best α=1000) | +0.28 | −0.78 | ~1.06 R² gap = drift signal |
| **S_feat off-diag mean** | 0.725 | **0.771** | Dreamer sees ep1 frames as *more uniform* than VGGT does |
| **S_deter off-diag mean** | 0.784 | **0.826** | Same, stronger in deter — collapse is *recurrent* |
| **S_VGGT off-diag mean** | 0.759 | 0.693 | VGGT correctly distinguishes ep1 frames more than ep7 (longer trajectory, more places visited) |
| **Pearson(S_VGGT, S_feat)** | 0.380 | 0.568 | *Inverted* — see below |

### The Pearson inversion: why "higher correlation" is misleading

A naive read of the Pearson row would conclude "ep1's Dreamer agrees with VGGT *more* than ep7's." That's wrong. The resolution comes from the off-diagonal means:

- VGGT distinguishes ep1 frames (mean cosine = 0.693, wide spread).
- Dreamer collapses ep1 frames (mean cosine = 0.771–0.826, narrow spread).
- Pearson is scale-invariant — it measures whether the **rank** of similarity is preserved.
- Both matrices preserve the same rank ordering, so Pearson is high.
- But Dreamer compresses the *amplitude* — different parts of the room look more alike in latent space than in VGGT space.

This is **representation collapse** in the classical sense: relational topology preserved, absolute geometric resolution lost.

The matching trajectory variability (`feat.std` over time = 0.165 on ep1 vs 0.177 on ep7) rules out a stationarity artifact — Dreamer isn't just stuck, it's actively flattening.

### What this says about the thesis question

The encoder is not failing in a single localized way. It is **lossy in a specific direction**: it preserves *which* moments are similar to *which* others (relational structure), but the absolute geometric content of each moment is degraded — and the degradation is worse during the trajectories that fail to terminate. The recurrent (`deter`) component carries this collapse more strongly than the stochastic (`stoch`) component does, which is consistent with the prior callchain finding that `deter` underutilizes its capacity ([#90](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/90)).

A working hypothesis to verify with the 4-panel notebook: ep1 ends 0.94 m from the goal — 4.7× outside the 0.2 m success radius (`GOAL_RADIUS` in `habitat.py:23`). This is a **final-approach failure**, not a STOP-suppression bug. The question the notebook asks is whether the encoder's latent correctly represents that the agent is still far from the goal, or whether it has drifted to a "near-goal" representation early. If the similarity to early "approaching goal" frames is high at the final step, the encoder is wrong. **The visualization is the empirical answer.**

## Methodological surprises encountered (and parked as issues)

While running this analysis, three methodological problems surfaced that the thesis should be explicit about. All three are now GH-tracked:

- **[#101](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/101) — Eval is non-deterministic across re-runs on H100.** Two identical-seed identical-checkpoint runs produced different episode outcomes (ep11: 500-step failure vs 57-step success). Root cause: TF32 matmul drift accumulates over hundreds of matmuls until argmax flips. **Workaround**: pair-pick and dumps must come from the same run.
- **[#102](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/102) — Standalone evaluate.py was CNN-only until commit `9c8057d`.** No R2Dreamer + VGGT run had ever been evaluated through the standalone eval pipeline before this work; pre-`9c8057d` eval numbers in wiki experiment pages are trainer-internal eval, not standalone.
- **No GH issue, just a note**: linear probes are at the edge of usability for this latent. R² = 0.28 on the in-distribution success episode is barely above chance; ridge with α=1000 was needed to prevent extrapolation explosions on holdout. An MLP probe would likely raise absolute R² but the **gap** between ep7 and ep1 (the actual drift signal) is unlikely to shrink — this was the call to keep linear probes for now.

## Limitations

- **Probe absolute R² ≤ 0.28** on the best probe + episode. The *differential* R² is the load-bearing signal; absolute reconstructions are illustrative only.
- **Euclidean distance fallback** for end-to-goal computation (Habitat's `geodesic_distance` is not exposed by `evaluate.py`). For the L1 single-room scene this is a tight proxy. The 1.5 m near-miss threshold used in pair-picking is much larger than the formal 0.2 m success radius (`GOAL_RADIUS` in `habitat.py`); episodes flagged NEAR-MISS include trajectories that end well short of the goal.
- **Sample size n = 1 per regime** (one success, one near-miss). The per-episode findings could be idiosyncratic to the specific trajectory rather than representative of the policy. Replicating with a second pair (e.g. ep10 + ep0) would strengthen the claim — deferred.
- **Linear probe is OOD-fragile.** ep1's latent regime may genuinely fall outside what 6 LOST + 4 short-SUCCESS train episodes cover.

## Status & next steps

| Phase | Status |
|---|---|
| Patch standalone eval to support VGGT (commit `9c8057d`) | done |
| 20-episode pair-pick roll (`eval-pick-20ep/`) | done |
| Per-step instrumented re-roll (`viz-pair-a/`, 12 eps, 1.64 GB) | done |
| Linear ridge probes + α-sweep | done |
| Temporal similarity matrices C2 + spatial probe-error timeline C1 | done |
| Interactive 4-panel notebook (Plotly 3D cloud + trajectory + heatmaps) | done — [`notebooks/debug_viz_l1.ipynb`](../../../notebooks/debug_viz_l1.ipynb) |
| MLP probe escalation | deferred (Plan B if a reviewer pushes back on linear-probe absolute R²) |
| Replication on a second pair | deferred |

The notebook is a thin viewer over pre-rendered artifacts; loading the sparse-frame bundles (`output/methods/debug_viz/l1/notebook/bundle_ep00{1,7}.npz`) keeps it fast to open. Generated by `scripts/debug_viz/make_notebook.py` and `scripts/debug_viz/render_notebook_data.py`.

## Artifacts

- `output/runs/r2dreamer-curriculum-l1-vggt/baseline-actent3e-4/debug/eval-pick-20ep/` — 20-ep pair-pick (eval_results, MANIFEST, SUMMARY)
- `output/runs/r2dreamer-curriculum-l1-vggt/baseline-actent3e-4/debug/viz-pair-a/` — 12-ep instrumented dumps (`episode_NNN/step_TTT.npz`, MANIFEST)
- `output/runs/r2dreamer-curriculum-l1-vggt/baseline-actent3e-4/debug/viz-pair-a-attempt1/` — first re-roll, preserved for forensic comparison with attempt2 (used in the determinism investigation that produced #101)
- `output/methods/debug_viz/l1/probes/` — probe weights (`.npz`), per-episode predictions, SUMMARY
- `output/methods/debug_viz/l1/similarity/ep_001/`, `ep_007/` — `similarity.png` (5-panel C2), `similarity.npz` (raw matrices), `probe_error_timeline.png`

## Commits

| SHA | Description |
|---|---|
| `9c8057d` | feat(eval): support VGGT encoder in standalone evaluate.py |
| `d96b459` | feat(debug-viz): add per-step instrumented eval script |
| `1dccac7` | feat(debug-viz): linear probes feat/deter/stoch → world_points (ridge) |
| `d64a647` | feat(debug-viz): build C2 temporal similarity matrices + C1 probe-error timelines |

## Related

- [VGGT → R2Dreamer Call Chain](vggt-r2dreamer-callchain.md) — the upstream data flow + 5 high-leverage breakpoints; the foundation this work builds on.
- [Cross-Correlation Matrix](cross-correlation-matrix.md) — Barlow-Twins regularizer used in the encoder; relevant context for why a linear probe is even plausible.
