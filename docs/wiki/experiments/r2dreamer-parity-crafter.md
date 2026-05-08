---
run_path: output/runs/r2dreamer-parity-crafter/
slurm_id_internal: 4463832
slurm_id_external: 4464615
wandb_project: r2dreamer-parity-crafter
wandb_id_internal: TBD
wandb_id_external: TBD
status: running
---

# R2-Dreamer Parity — Internal (JAX) vs External (PyTorch ref) on Crafter

**Status**: running
**Date**: 2026-05-08
**Tags**: #parity #r2dreamer #crafter #ab #internal-jax #external-torch
**Wandb project**: [r2dreamer-parity-crafter](https://wandb.ai/sailer-luca-university-ulm/r2dreamer-parity-crafter)

## Setup

A/B comparison of two R2-Dreamer codebases on Crafter, isolated in a fresh
W&B project so the curves stay separate from the Habitat-flavored
`3d-vla-objectnav` work.

| Run | Impl | Source | Encoder | Logging path |
|---|---|---|---|---|
| Internal | `modules/r2dreamer` (JAX, our reimpl) | feat/vggt-film-encoder-109 branch | CNN (only Crafter-compatible option) | native `wandb.init` in `trainer.py` |
| External | `external/r2dreamer` (PyTorch ref) | NM512 ICLR 2026 — vendored upstream | size12M model, `rep_loss=r2dreamer` | TB → W&B via `run_external_crafter.py` wrapper (sync_tensorboard=True; **does not edit external/**) |

### Hyperparameter alignment

Variables under test (the *point* of the comparison): model architecture,
representation loss formulation, optimizer / scheduler details. Everything
else is matched.

| Knob | Internal | External | Status |
|---|---|---|---|
| Steps | `--steps 1000000` (overriding 2.4M default) | `env=crafter` → `1.01e6` (upstream default) | aligned within 1% |
| Batch size | 16 | 16 | aligned |
| Sequence / batch length | 64 | 64 | aligned |
| Image size | (3, 64, 64) | (64, 64) | aligned |
| Action repeat | 1 (Crafter env wrapper) | 1 (`crafter.yaml`) | aligned |
| Prefill | 5000 | upstream default | aligned |
| Seed | 0 | 0 | aligned |

### W&B run names

- Internal: `internal-jax-crafter-cnn-s0-${SLURM_JOB_ID}`
- External: `external-torch-crafter-ref-s0-${SLURM_JOB_ID}`

### Reproduction

From repo root on BWUniCluster (login node):

```bash
# one-time external venv (~5–10 min, ~3 GB)
bash modules/r2dreamer/scripts/setup_external_venv.sh

# kick off A/B
sbatch modules/r2dreamer/scripts/slurm/ab_internal_crafter_s0.sbatch
sbatch modules/r2dreamer/scripts/slurm/ab_external_crafter_s0.sbatch
```

After both finish, fill in the SLURM and W&B run IDs in the frontmatter
above and update `status` to `done`.

## Hypothesis

Our JAX reimplementation should reach Crafter-score parity with the
upstream PyTorch reference (within ±2 absolute Crafter-score points) by
1M env steps. A larger gap → root-cause investigation; matched curves →
mark `status: blessed` and document.

## Open caveats

- **Model size mismatch**: the internal impl uses a smaller default world
  model than upstream's `size12M`. We deliberately compare the
  *as-defaulted* configurations because that's how each codebase is
  used in practice. A second-pass A/B at matched parameter counts is
  out of scope for this smoke A/B (1 seed each).
- **Single seed**: not statistically powered. This is a parity smoke,
  not a publishable claim. If results agree, scale to 3–5 seeds before
  publishing.
- **Eval cadence**: external evals every 10k steps with 10 episodes
  (`crafter.yaml`). Internal does not run periodic eval episodes by
  default — `eval/` series will be sparse on the internal side. Compare
  `train/episode_score` (or equivalent) for matched-density curves.
- **Logging dialect**: scalar names differ (e.g. external `episode/score`
  vs internal `train/episode_return`). The W&B dashboard will need
  per-impl panels until a name-mapping doc lands.

## Results

*Pending. Update with Crafter score, eval_score curves, and learning
dynamics once both runs complete.*
