# Step-time optimization loop — baseline fvwuoux3

Status: ready-for-agent (selection received 2026-07-07)

## Selected

1. **pipeline-full-bf16** (user-proposed): make the entire JAX pipeline run
   bfloat16 end-to-end — find and convert remaining float32 (params, obs
   transform, replay batch dtypes, house buffers), config-gated.
2. **replay-vectorized-storage**: preallocated contiguous arrays + windowed
   gather instead of per-transition deepcopy objects.

Deselected: vggt-extract-every-2, vggt-compile-fp16, vggt-resolution-down,
vggt-kv-budget-trim, replay-async-prefetch.

## Baseline

- Run: `sailer-luca-university-ulm/3d-vla-objectnav/fvwuoux3`
  (l1_hybrid_house_points_pose-5812468, state=running, 2M steps, batch 16, bf16)
- Measured 2026-07-07 via `fetch_step_time.py`, n=1800 post-warmup samples:
  **median 184.3 ms/step** (p10 177.8, p90 190.5, mean 184.0 ± 12.9)
- Attribution (from sibling prod-shape profile, job 5762154,
  `docs/notes/house-points-pose-profiling.md`): vggt_extract ~132 ms (62%),
  replay_sample ~59 ms amortized (27%), train_step ~20 ms (9%), agent_act +
  env_step + house glue ~7 ms. Hybrid run sits at 184 vs 219 for the pure
  house-points-pose shape; same loop structure, VGGT still the floor.

## Candidates proposed

| # | Name | Attacks | Expected | Risk |
|---|------|---------|----------|------|
| 1 | replay-vectorized-storage | replay_sample 59 ms | up to −50 ms (~27%) | batch-assembly correctness (episode boundaries, dtypes) |
| 2 | replay-async-prefetch | replay_sample 59 ms | −20–40 ms (overlap, bounded) | threading/ordering complexity |
| 3 | vggt-extract-every-2 | vggt_extract 132 ms | ~−60 ms at K=2 | staler points/pose → quality regression |
| 4 | vggt-resolution-down | vggt_extract 132 ms | −30–60 ms | coarser geometry, worse house map |
| 5 | vggt-compile-fp16 | vggt_extract 132 ms | −20–40 ms | compile warmup, numeric drift, KV-cache graph breaks |
| 6 | vggt-kv-budget-trim | vggt_extract attention | −10–25 ms | multi-frame consistency loss |

Prior art: `scripts/profiling/profile_jax_replay_buffer.py`, modal-replay
configs (replay side); `ReplayBuffer` object store with `deepcopy` per `add`
at `src/buffer/replay_buffer.py:251`.

## Results

| Variant | Commit | Job | ms/step | Δ vs 184.3 | Verdict |
|---|---|---|---|---|---|
| replay-vectorized-storage | 84e4b3b (pre-existing) | — | — | — | already-landed |
| control probe (unmodified) | branch feat/step-time-opt-fvwuoux3 | 5837512, 5838392 | — | — | failed (infra) |
| pipeline-full-bf16 (full probe) | 70e3311 | 5838214 | — | — | failed (infra) |
| pipeline-full-bf16 (train_step microbench) | e81997d | 5839014 | 44.82 (train_step) | −15.1 ms/call (−25.2%) | keep (gated, default off) |

### Result (H100, prod shape B=16 T=64, 262k house points, n=50)
- float32 train_step: 59.95 ms/call median (p10 59.4, p90 61.0)
- full_bf16 train_step: 44.82 ms/call median (p10 44.0, p90 50.0)
- **−15.13 ms/call (−25.2%)** on the JAX train_step.
- Env-step projection: at train_ratio 512 the profile amortized train_step at
  0.5 call/env-step, so ≈ **−7.6 ms/env-step, ~4% of the 184 ms baseline**.
  (Projection, not an end-to-end measurement — the full training probe that
  would confirm it is blocked by the venv/habitat infra issue above.)
- Verdict **keep, gated default-off**: real gain, ships dark safely. Loss-quality
  parity needs a full training run once the venv is repaired — recommended in PR,
  not flipped on here.

### Infra blocker: full training probes can't run
Both full-shape training probes died at env init, NOT in my code:
- Prod-mode `scripts/slurm/launch.sh` runs `uv run python`, which re-syncs the
  shared `.venv` ("Uninstalled 37 / Installed 37 packages"; venv had drifted —
  "missing RECORD file" warnings from out-of-band pip installs). The re-sync
  leaves habitat un-importable under the locked numpy 2.4.6
  (`habitat/utils/visualizations/maps.py:44` `.squeeze(1)` raises on a
  non-size-1 axis). The live baseline 5812468 survives only because it imported
  habitat before the churn. Repairing the shared venv (`uv sync` / numpy pin)
  is a destructive action on shared state with a 29h run live on it — NOT done
  unilaterally; flagged for the user.

### Pivot: isolate what full_bf16 actually changes
full_bf16 touches only the JAX model compute (encoders/RSSM/heads/optimizer),
never torch VGGT (132 ms) or the host replay sampler (59 ms). A full training
probe would bury the effect under those fixed costs anyway. So measure the JAX
`train_step` directly with a synthetic prod-shape batch — no habitat import,
`.venv/bin/python` directly (no uv sync). Script:
`scripts/profiling/profile_full_bf16_step.py` (+ .sbatch), job 5839014.
CPU tiny-shape smoke already shows the mechanism: the 262k-point house MLP
dominates train_step and bf16 ~halves it (-45% train_step on CPU).

### Notes
- **full_bf16 implemented** (70e3311): cfg.full_bf16 gate threads compute_dtype
  through ConvEncoder / HousePointsCameraEncoder+Hybrid / Deter+BlockLinear /
  R2RSSM / R2MLP. float32 master params, float32-pinned RSSM+head logits,
  float32 RMSNorm stats, GNN float32 exemption kept. CPU verified: default path
  62 tests pass; gate smoke confirms encoder emits bf16 on / float32 off, both
  finite over 4 train steps, params stay float32, act() works.
- **Audit finding:** compute_dtype was a near no-op outside the token
  transformer — CNN/house/pose encoders hard-cast inputs to float32. So the
  baseline fvwuoux3 (bfloat16 config) was really training float32 end-to-end;
  full_bf16 is the first real reduced-precision path for this encoder family.
- **Worktree infra fix:** first control probe (5837512) crashed on habitat
  import — the worktree had broken *local* .venv/data/output dirs instead of
  symlinks to the main checkout. Fixed via scripts/setup_worktree.sh; both
  probes re-launched (control 5838392, bf16 5838214).

- **replay-vectorized-storage: already implemented on main.**
  `src/buffer/replay_buffer.py` was converted to structure-of-arrays ring
  storage with vectorized fancy-index gather on 2026-07-04 (84e4b3b,
  "finish ReplayBatch attribute migration + buffer SoA storage") — citing
  the exact 119 ms/batch profiling finding this candidate targeted. The
  baseline run fvwuoux3 (started 2026-07-06 from 3a86501, descendant of
  84e4b3b) already contains it. The 219→184 ms drop vs the 07-04 profile
  is consistent with this win (attribution imperfect: encoder also changed
  house-points-pose → hybrid). No further work; candidate closed.
