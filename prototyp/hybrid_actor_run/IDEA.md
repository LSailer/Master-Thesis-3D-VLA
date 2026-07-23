# Hybrid actor run — IDEA

## Goal

Run the **hybrid** pipeline end-to-end with the **trained Dreamer actor**
instead of a random agent, and export the cumulative downsampled house
cloud it builds along the way.

"Hybrid" here means the production `vggt_hybrid_house_points_pose` setup:
per step the actor sees

- the current 64×64 RGB frame (CNN backbone branch),
- the current VGGT camera pose (9-dim, gated MLP branch),
- the **cumulative, voxel-deduped house point cloud** accumulated live from
  VGGT world points across the whole run (gated PointNet branch).

This is the actor-driven successor of
`prototyp/live_vggt/run_house_visualization.py`, which walked randomly and
accumulated points host-side.

## Key understanding (why the run file is thin)

Everything the prototype needs already exists in production code:

- `encoder_registry["vggt_hybrid_house_points_pose"]`
  (`src/r2dreamer/encoders/house_points_pose.py`) builds the JAX VGGT
  extractor in `PERSIST_SCENE` mode and wraps it in
  `VGGTHybridHousePointsPoseObsAdapter`
  (`src/r2dreamer/adapters/house_points_adapter.py`).
- The adapter's `prepare_env_step(obs)` runs VGGT once per frame, feeds
  every confident point into a per-scene `HouseContextPoseBuffer`
  (device-side voxel dedup, default 0.01 VGGT units — the cloud **is
  already downsampled**, no end-of-run `voxel_down_sample` needed), and
  emits the fixed-size `(max_points, 6)` snapshot + valid count the
  encoder pools over.
- `launch/evaluate.py` shows the exact actor loop:
  `R2DreamerAgent.from_checkpoint` with architecture overrides read from
  the training MANIFEST.json, then
  `agent.act_with_state(encoder_obs, is_first, act_state, key)` per step.
- `buffer.save(dir)` exports the accumulated cloud as binary PLY.

So `run_hybrid_actor.py` only orchestrates: eval env (518×518 HWC) →
adapter → actor → per-episode metrics → final PLY/JSON/MP4 export, with
MANIFEST.json as the run-status ground truth.

## Hypothesis / what we want to see

- The actor's task-directed trajectories produce a house cloud whose
  coverage differs from random walks (likely more focused near goal
  routes, less complete overall). Comparing `house_cloud/*.ply` +
  `growth_history` against live_vggt random-walk runs quantifies this.
- Sanity check that the live hybrid observation contract
  (image + camera_pose + house context snapshot) round-trips through
  `act_with_state` outside the trainer, i.e. eval-time behaviour of the
  hybrid encoder is healthy.

## Planned approach

1. `run_hybrid_actor.py --checkpoint <ckpt> --episodes 5` on a GPU node
   (SLURM); `--random` as plumbing smoke test without a checkpoint.
2. Judge success via MANIFEST.json status, `results.json`
   (success/SPL/steps + `house_buffer` diagnostics + `growth_history`),
   and the exported `house_cloud/<scene>/step_00000_context.ply`.
3. Later: side-by-side cloud comparison actor vs. random (same scene,
   same step budget).
