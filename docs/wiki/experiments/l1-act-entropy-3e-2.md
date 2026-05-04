---
run_path: output/r2dreamer-curriculum-l1/run-<slurm_job_id>/
slurm_id: TBD
wandb_id: TBD
status: running
---

# L1 Rerun — act_entropy=3e-2 Baseline Restore

**Status**: running (submitted to unicluster 2026-04-30)
**Branch**: `experiment/act-entropy-3e-2`
**Tags**: #baseline-restore #act-entropy #l1 #r2dreamer #habitat
**Wandb tag group**: `act-entropy-3e-2,baseline-restore,l1`

## Motivation

The confirmed 75% SR / 0.55 SPL result ([l1-rerun-buffix, wandb y5a0upzd](l1-rerun-buffix.md)) was
trained with `act_entropy=3e-2` (the previous hardcoded default). Commit 9b5f26f reverted to the
DreamerV3 paper default (`3e-4`) based on the theoretical argument that `α·H(π) ≈ 0.042` was ~100×
too large. However, the lower-entropy baseline has not yet reproduced 75% SR.

This run re-establishes whether 3e-2 is necessary to reach the 75% SR level on L1, with everything
else held equal to the blessed l1-rerun-buffix run.

**Hypothesis**: `act_entropy=3e-2` is load-bearing for the 75% SR result. The higher entropy keeps
the policy from collapsing to near-deterministic navigation before the world model has converged.

## What changed vs the blessed run (y5a0upzd)

| | This run | Blessed (y5a0upzd) |
|---|---|---|
| `act_entropy` | `3e-2` (explicit CLI) | `3e-2` (old hardcoded default) |
| Config default | `3e-4` (unchanged on main) | n/a |
| Buffer fix | yes | yes |
| Step penalty | -0.01 | -0.01 |
| Max episode steps | 1000 | 1000 |

Everything else is identical. The `--act_entropy 3e-2` flag is now explicit in the sbatch script
(`modules/r2dreamer/scripts/slurm/train_curriculum_l1.sbatch`) so the setting is reproducible
independent of the config default.

## Configuration

| Parameter | Value |
|-----------|-------|
| total_steps | 2,400,000 |
| act_entropy | **3e-2** |
| batch_size | 16 |
| seq_len | 64 |
| train_ratio | 512 |
| imagination_horizon | 15 |
| lr | 4e-05 |
| deter_size | 2048 |
| stoch_discrete | 16 × 32 |
| step_penalty | -0.01 |
| success_bonus | 10 |
| max_episode_steps | 1000 |
| buffer_capacity | 1,000,000 |
| scene | fK2vEV32Lag (1 house, chair only) |

## Results

*Run in progress — update once SLURM job completes.*

| Metric | This run | Blessed (y5a0upzd) | 3e-4 baseline |
|--------|----------|-------------------|---------------|
| Success Rate | TBD | 75% | TBD |
| SPL | TBD | 0.55 | TBD |

## Interpretation template

Fill in once complete:

- If SR ≈ 75%: confirms `act_entropy=3e-2` is the right value; proceed to encoder ablation with this as the fixed baseline.
- If SR < 70%: something else changed between runs (check seed, data shuffle, curriculum episode count). Do not flip entropy back to 3e-4 until root-caused.
- If SR > 75%: unlikely but possible if seed is luckier; still use this as the baseline.

## Related

- [l1-rerun-buffix](l1-rerun-buffix.md) — the blessed 75% SR run this attempts to restore
- [encoder-fusion-plan](../methods/encoder-fusion-plan.md) — ablation matrix gated on this baseline
