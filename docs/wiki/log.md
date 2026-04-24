# Wiki Log

## [2026-04-20] update | torch.compile spike on VGGT | source: /plan Slice 3 follow-up (PRD #74)
`VGGTFeatureExtractor(compile=True)` wraps aggregator/camera_head with dynamic=True, point_head static. Measured 11% p50 reduction on vggt_forward (168.3 → 149.8 ms), p95 also 11%, more consistent frame-to-frame. Per-step p50: 190.7 → 171.0 ms. Shipped as opt-in flag + `--compile` CLI. Wiki section appended to methods/l4-profiling.md. JSON: output/profiling/vggt_vs_cnn_20260420_120455.json.

## [2026-04-20] ingest | L4 Pipeline Profiling (VGGT vs CNN) | source: /plan Slice 3 (PRD #74)
Full 2000/2000 diagnostic on H100. VGGT forward = 168.3 ms p50 — 90% of the VGGT-vs-CNN slowdown. PyTorch↔JAX boundary (vggt_wrapper + jax_upload) = 0.44 ms — disproves the original #72 motivation. KV-cache reset contract verified (9 resets, 9 boundaries). Recommendation: repurpose #72 away from JAX port toward torch.compile / frame-skip / distillation. Created methods/l4-profiling.md. JSON: output/profiling/vggt_vs_cnn_20260420_{102405,103840}.json.

## [2026-04-16] ingest | Curriculum Scaling (L1-rerun, L2, L3) | source: /reporter
Updated experiments/l1-rerun-buffix.md, experiments/l2-1house-6goals.md, experiments/l3-10houses-chair.md with completed status. Semantic floor plan data collected for fK2vEV32Lag (L2 scene). Plot scripts and combined slide deck created.

## [2026-04-16] ingest | L1 Rerun, L2, L3 Experiments | source: /reporter (paused)
Three curriculum experiments completed (SLURM 3957651, 3957713, 3957714). L1 rerun: 75% SR (buffer fix +8pp). L2: 36% avg SR with goal hierarchy (plant 66% > tv_monitor 3%) — Geo/Euc ratio is strongest predictor. L3: 32% SR across 10 houses. Created experiments/l1-rerun-buffix.md, l2-1house-6goals.md, l3-10houses-chair.md. Reporter paused pending semantic floor plan rendering on cluster (issue #71). Updated index.

## [2026-04-15] ingest | Training Orchestration | source: /engineer
Implemented RFC #68: unified ReplayBuffer+VGGTReplayBuffer with BufferConfig, created Trainer module (convert_batch, checkpoint with ema_state fix, ObsAdapter, habitat_defaults). Rewrote 3 training scripts (-62% lines). Created methods/training-orchestration.md. Updated index.

## [2026-04-15] ingest | World Model Training Loop | source: /plan interview
Explained how 64-step replay windows, 15-step imagination horizon, and full-length acting episodes fit together. Created methods/world-model-training-loop.md. Updated index.

## [2026-04-15] ingest | L1 Baseline 2.4M | source: /reporter
R2-Dreamer on L1 curriculum (wandb krokhgwi, SLURM 3923812). 67% SR, 0.49 SPL — 17x above random. World model learns but overfits (val dyn 17→42). Created experiments/l1-baseline-2.4m.md. Updated index. Generated 5 plot figures.

## [2026-04-13] ingest | Random Baseline L1 | source: /reporter
Random agent on L1 curriculum (834 eval episodes, 1 house, chair only). 3.84% SR, 0.023 SPL, -4.40 mean reward. Created experiments/random-baseline-l1.md. Updated index. Generated 4 plot figures.

## [2026-04-13] ingest | Baseline 2.4M All Scenes | source: /reporter
Analyzed r2dreamer-baseline-2.4M-3907457 (wandb qwdqowxq). 4871 episodes, 2.36% SR (random-level), no learning. Created experiments/baseline-2.4m-all-scenes.md. Updated index. Generated 3 plot figures.

## [2026-04-13] ingest | 2026-03-03 Braun Meeting | source: manual migration
Migrated from docs/notes/2026-03-03-meeting.md. Created meetings/2026-03-03-braun.md. Updated index.

## [2026-04-13] ingest | Claude Code Workflow | source: manual
Created methods/workflow.md documenting full pipeline, skill inventory, TDD rules, wiki conventions. Updated index.
