# Wiki Log

## [2026-04-24] perf | VGGT JAX camera_head jit (closes #81) | source: /grill-me + direct implementation
Camera_head was the last eager phase in the JAX extractor: 1187 ms/frame (95% of extract). Mirrored PR #80's padded 3-tuple + `jax.jit` pattern: single compiled graph (no `is_first_frame` static arg needed — `valid_len=0` covers frame 0 in the padded path), `_CAM_MAX = max_camera_frames × num_iterations = 4096` slots, no eviction (camera cache holds the whole episode). Added `test_jax_parity.py::TestLevel3CameraHeadPaddedParity` locking padded-vs-legacy bit parity (atol 1e-5) and `test_jax_integration.py::test_camera_cache_overflow_raises` for the overflow guard (raises `RuntimeError` instead of silent corruption via `dynamic_update_slice` clamping). Results (bf16, H100, n=10): camera_head 1187 → 4.33 ms (274×); extract total 1247 → 64 ms; `bench_fast` JAX median 1763 → 137 ms vs PT 72 ms → speedup 0.041 → **0.524×** (issue exit gate 0.5× met). Parity bit-identical to pre-change on L1 episode 23795 dump. Bug caught mid-session: `profile_streaming.py` was calling `ext._camera_head.apply` eagerly, bypassing the new jit wrapper — first post-change profile showed no improvement until the bench was fixed to route through `ext._camera_head_apply`. Commits: 90e123d (code+tests+dump script) + 648abff (tsv hash fix). Created `methods/vggt-jax-streaming.md`. Phase 2 (shared padded_cache helper extraction) parked for a future session.

## [2026-04-24] refactor | dreamerv3 archival + architecture audit | source: /grill-me + /improve-codebase-architecture
Cleanup session. Archived `modules/dreamerv3/` as-is to `archiv/dreamerv3-20260424/` via `git mv`; peeled `configs.py`, `optim.py`, `replay_buffer.py` + `test_replay_buffer.py` to `modules/shared/`; updated 17 import sites; `pyproject.toml` testpaths updated. 90/90 tests pass (76 shared+r2dreamer, 14 envs). Output dir 37 GB → 21 GB (deleted 6 stale run dirs + 3 checkpoint dirs from `r2dreamer-habitat-baseline/`, kept metrics.csv + slurm logs). Architecture audit produced 3 deepening RFCs with parallel sub-agent design rounds: #84 (deepen `HabitatObjectNavEnv` — owned types + thin NavBackend port + stub), #85 (deepen training entrypoint — two-axis registry + curriculum dataclass, supersedes #52), #83 parked (base `WorldModelConfig` consolidation). Perf brainstorm parked as #82 for next session. Closed #69. Grill surfaced that L1-VGGT job 3957756 already reached ~47-63% SR (resume artifact explains dip) — below 2D L1 (75%) but not a blocker since curriculum-level comparison (L2/L3/L4) is the real plot. Research direction locked: primary plot = 2D-CNN vs 3D-VGGT curriculum SR; encoder = raw VGGT first, semantic DPT head as stretch; UNITE dropped.

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
