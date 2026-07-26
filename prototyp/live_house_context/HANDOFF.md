# Handoff — live house context PERSIST

## State (2026-07-03) — DONE, goal met

- **Fix implemented & verified.** Scene-aware `on_episode_reset(scene_id)` at
  all 4 trainer reset sites (prefill start, prefill episode-end, train reset,
  eval reset) + `evaluate.py`; adapter lambdas `lambda scene_id="scene":
  extractor.reset_for_scene(scene_id)` (backward-compat default);
  `obs_adapter.py` signature updated. The prefill loop now captures the reset
  frame it previously discarded, so `reset_for_scene` fires during prefill and
  the prefill frame carries into train (the PROTOCOL §2 root cause is fixed).
- **Smoke: PASS** (job 5738777, `house_points_pose_l1_live --smoke`, 600+600).
  Canonical `=== Smoke PASS ===`, 235 metric rows. Growth dropped 3.77 M to
  2.52 M (−33%); prefill-to-train acceleration gone (+0.79 M vs +2.0 M pre-fix).
- **PERSIST frame preservation: PASS** (clean extractor-replay test, job
  5739251). Mean tail IoU (continuous vs save+reset_for_scene+restore) =
  **1.000**, NN median 2–6 mm. Assumption A2 confirmed: the restored VGGT cache
  preserves the world frame. (The first, episode-based diagnostic at 5739050
  reported FAIL but was **confounded** — different episodes per `env.reset()`;
  see PROTOCOL §7.4. Redesigned to the replay test.)
- **CPU regression: zero regressions** — 69 passed, 2 skipped, 1 pre-existing
  failure (`test_reset_train_episode_uses_prepare_env_step_when_available`,
  verified pre-existing via `git stash`; stale `packer`-contract test, not a
  PERSIST issue). Reset + encoder-config suite: 22 passed.

## Next step — submit the prod run

The full pipeline is verified. To launch the 2M-step prod run:
```bash
bash scripts/slurm/launch.sh house_points_pose_l1_live --prod
```
(or `--smoke-then-prod` for an afterok-gated auto-chain).

## Known follow-ups (not blockers for the prod run)

- **Pre-existing stale test** `test_reset_train_episode_uses_prepare_env_step`
  (mock expects a 2-arg `prepare_env_step(env_obs, packer)` the trainer never
  had). Separate cleanup.
- **bf16 `store_xyz` aliasing beyond ~2.56 m** — pre-existing; revisit only if
  the prod run's end-of-run geometry summary shows residual far-point noise.
- **Camera-head eviction cost (R5)** — watch step time vs the 158 ms baseline in
  the prod run; lower `max_camera_frames` if it regresses once the cache fills.

## Files

- **Fix (production code):** `src/r2dreamer/adapters/{obs_adapter,vggt_adapter,
  hybrid_adapter}.py`, `src/r2dreamer/trainer.py`, `src/r2dreamer/launch/evaluate.py`,
  `src/vggt/jax/feature_extractor.py` (in-extract is_first to reset_for_scene).
- **Tests:** `tests/vggt/test_reset_for_scene.py`,
  `tests/r2dreamer/launch/test_encoders.py` (PERSIST + scene-aware callback).
- **Protocol (this folder):** `IDEA.md`, `PROTOCOL.md`, `PROBLEMS.md`,
  `HANDOFF.md`.
- **Diagnostic (removed):** `check_persist_alignment.py` and its launcher
  `run_alignment.sbatch` were deleted in the adapter-routing refactor - they
  targeted `src.r2dreamer.encoders.base` and `src.r2dreamer.launch.habitat_setup`,
  neither of which survived it. What they measured and found is preserved in
  PROTOCOL §7.

## Reproduce the verification

```bash
# 1. CPU unit tests
JAX_PLATFORMS=cpu uv run pytest tests/vggt/test_reset_for_scene.py \
    "tests/r2dreamer/launch/test_encoders.py::TestVGGTEncoderConfiguration" -q

# 2. PERSIST frame-preservation diagnostic — no longer reproducible: the
#    script and its sbatch wrapper were removed in the adapter-routing
#    refactor. Its verdict is recorded in PROTOCOL §7.

# 3. Full-pipeline smoke (GPU, ~8 min, 30-min cap)
bash scripts/slurm/launch.sh house_points_pose_l1_live --smoke
```