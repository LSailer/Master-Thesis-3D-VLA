# Archived PyTorch r2dreamer entrypoints

Archived 2026-04-25 as part of #85 launcher refactor (Phase 3).

These scripts wrapped the PyTorch reference implementation at `external/r2dreamer/`.
They served two purposes during the JAX port: (1) running comparison training jobs,
(2) parity-validating outputs against the JAX implementation.

After the JAX path stabilized (closing #81 — VGGT JAX streaming) and the thesis
pivoted to world-model 3D-vs-2D comparisons, these scripts became dormant:

- Audit (2026-04-25, in #85 grill session): not invoked by any active curriculum
  sbatch, not cited as the source of any wiki experiment, last meaningful change
  2026-03-29 (creation/move only).
- The active parity flow now runs through `modules/r2dreamer/launch/parity/`.

Preserved here for parity-history and reproducibility reference. Do not import.
