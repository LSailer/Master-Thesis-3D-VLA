# ADR 0002 — No frame-skip in training; accept the 4-day chained L4 cycle

**Date**: 2026-05-04
**Status**: Accepted
**Supersedes**: implicit assumption in issue #97 (frame-skip as throughput lever)

## Context

The thesis tests *whether R2Dreamer performs better with 3D (VGGT) vs 2D (CNN) features on HM3D ObjectNav* (`CLAUDE.md`). The headline plot is the L4 curriculum SR comparison between a VGGT-encoded agent and a CNN-encoded agent.

Per `docs/wiki/methods/l4-profiling.md`, the L4-VGGT path runs at ~5.2 FPS on H100, vs ~30 FPS for L4-CNN. A single SLURM job is capped at 48h on uc3 (verified: job `4109486`, killed at exactly 2-00:00:28). Thesis-relevant step counts (~3.2 M env steps) therefore require **chained jobs across ~4 days of wall time** for VGGT, against ~24 h for CNN.

`l4-profiling.md` proposed frame-skip (#97) as the obvious throughput lever — run VGGT every K env steps, reuse cached features in between. K=2 ≈ 2× throughput on the VGGT side, taking the cycle from ~4 days to ~2 days.

We considered frame-skip and rejected it.

## Decision

**Neither thesis comparison runs nor iteration runs use frame-skip.** Both arms (VGGT and CNN) run the encoder on every env step. The 4-day chained L4-VGGT cycle is accepted as the production cost.

Iteration testing is done at smaller curriculum levels (L1 / L2), which already complete in single-job time without frame-skip.

## Considered alternatives

1. **Frame-skip in both arms (K=2 VGGT and K=2 CNN).** Apples-to-apples but biases the comparison: CNN gains nothing from skip (already 30 FPS), and the agent's effective control rate halves. Changes what's being measured.
2. **Frame-skip in the VGGT arm only.** Breaks comparison cleanness — the two arms now process inputs at different rates.
3. **Frame-skip in *iteration* runs only, not thesis runs.** Surface-attractive but undermines the iteration test's purpose: a code change that passes the K=2 iteration smoke can still break the K=1 thesis pipeline. Also requires maintaining two configs.
4. **Frame-skip with RGB fallback for skipped steps.** Mixes 2D and 3D inputs in the 3D arm. Kills the thesis claim. Rejected for thesis integrity.

## Rationale

The thesis claim is *structural* — "3D-only encoder vs 2D-only encoder" — and is strongest when both arms see exactly the same observation cadence. Frame-skip in any form introduces a confound: stale-feature dynamics (which only the VGGT arm has to bridge), or a 2D fallback that contaminates the 3D arm.

Iteration speed is recovered through orthogonal levers that don't change what's being compared:
- L1/L2 sanity runs for fast iteration (already <1 day).
- `torch.compile` mode tuning (Option B in `docs/wiki/methods/vggt-jax-eviction-recompile.md`'s fix list — separate work).
- The JAX uniformity track (#98) for downstream trainable-head work.

The 4-day production cycle is documented as expected and chained-job tooling is the supported path.

## Consequences

- **Production** L4-VGGT runs are scheduled as 2× chained 48h SLURM jobs (or 4× chained 24h jobs), resuming from checkpoint. The Trainer module's checkpoint contract (`modules/r2dreamer/scripts/profile_training.py`-adjacent) must reliably resume.
- **Issue #97** (frame-skip) is closed `wontfix` with this ADR cited.
- **Future research** that wants frame-skip — e.g. a follow-up project on async perception in DRL — can revisit this decision; it is scoped to the current thesis claim.
- **Speed work continues** but only along thesis-safe levers: compile-mode tuning, JAX uniformity (#98), and (out-of-scope for this ADR) any approved method-change like distillation, which would require its own ADR.
