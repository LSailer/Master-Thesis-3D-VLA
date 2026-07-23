# Hybrid actor run — HANDOFF

_Last updated: 2026-07-23_

## State

- `run_hybrid_actor.py` written, **not yet executed** (needs GPU node +
  checkpoint). Modeled on `src/r2dreamer/launch/evaluate.py` (actor loop,
  manifest arch overrides) + `prototyp/live_vggt/run_house_visualization.py`
  (manifest status, video/PLY export conventions).
- IDEA.md documents the pipeline understanding; PROBLEMS.md the open
  risks (HWC checkpoint cutoff, MANIFEST-next-to-checkpoint requirement).

## Next steps (in order)

1. **Pick a checkpoint**: a post-2026-07-22 (HWC)
   `vggt_hybrid_house_points_pose` training run with MANIFEST.json next to
   (or one dir above) the checkpoint path. Candidate: the live hybrid run
   `fvwuoux3` family — locate its output dir under `outputs/`.
2. **Plumbing smoke test** (no checkpoint needed), on a GPU node:
   ```
   .venv/bin/python prototyp/hybrid_actor_run/run_hybrid_actor.py \
       --random --episodes 1 --max-steps 30 --video-fps 0
   ```
   Use `.venv/bin/python` directly (not `uv run` — it can re-sync and
   corrupt the shared venv). Submit via `scripts/slurm/` conventions or
   `srun` with 1 GPU; login node is CPU-only.
3. **Actor run**:
   ```
   .venv/bin/python prototyp/hybrid_actor_run/run_hybrid_actor.py \
       --checkpoint <ckpt_dir> --episodes 5 --seed 42
   ```
4. **Judge the run** by `<run_dir>/MANIFEST.json` status == "ok" (NOT the
   SLURM exit code), then inspect `results.json` (success/SPL/steps,
   `house_buffer` diagnostics, `growth_history`) and
   `house_cloud/<scene>/step_00000_context.ply`.
5. Compare the actor cloud vs. a random-walk cloud (same scene/steps,
   `--random`) — coverage/point-count; consider reusing
   `prototyp/live_vggt/point_change_plot.py` ideas for the comparison.

## Gotchas for the next session

- Episode boundaries: only `env.reset()` + `adapter.on_episode_reset(scene_id)`.
  Never call `extractor.reset()` manually — PERSIST_SCENE saves/restores
  the per-scene VGGT cache; a manual reset would re-anchor the world frame.
- The house cloud in the buffer is already voxel-deduped on device
  (0.01 VGGT units, adapter default) — no post-hoc downsampling step.
- `--encoder vggt_house_points_pose` also works (no CNN branch) if the
  checkpoint was trained with that encoder; the run file hard-fails on an
  encoder/checkpoint mismatch.
- Outputs go to `outputs/prototype/hybrid_actor_run/<run_tag>/` per the
  prototyp workspace rules.
