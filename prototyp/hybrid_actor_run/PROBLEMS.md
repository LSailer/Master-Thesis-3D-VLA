# Hybrid actor run — PROBLEMS / open questions

- **Checkpoint compatibility (HWC):** the repo switched to a repo-wide HWC
  image contract on 2026-07-22; checkpoints trained before that are
  incompatible. Only use post-switch `vggt_hybrid_house_points_pose`
  checkpoints.
- **MANIFEST.json required next to the checkpoint:** architecture
  overrides (deter/stoch sizes, house_point_norm, …) come from the
  training run's MANIFEST.json via
  `evaluate._load_arch_overrides_from_manifest`. A bare checkpoint dir
  without a manifest builds the agent with config defaults → likely
  restore/shape errors.
- **Private-API usage:** the final export reads `adapter._buffers`
  (private) and the run file imports the private helper
  `_load_arch_overrides_from_manifest` from `launch/evaluate.py`.
  Fine for a throwaway prototype; graduate properly if this becomes real.
- **Multi-scene curricula:** the house buffer is per-scene, but the
  adapter's replay-batch injection assumes single-scene (exact only on
  L1). For this eval-only prototype that only affects interpretation if a
  curriculum with >1 scene is used — clouds are still per-scene correct.
- **Untested as of 2026-07-23:** the run file has not been executed yet
  (needs a GPU node + a valid hybrid checkpoint). See HANDOFF.md.
- **Cluster hazards:** judge runs by MANIFEST.json status, not SLURM exit
  code (habitat GL teardown poisons exit codes). Avoid node `uc3n089`
  (aborts habitat GL sensor reads, exit 134).
