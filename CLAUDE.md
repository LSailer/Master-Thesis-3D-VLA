# CLAUDE.md

## Project Overview

Master's thesis: **World Models + 3D Scene Understanding** — testing whether world models (e.g. DreamerV3) perform better with 3D semantic features (UNITE) vs 2D-only for object navigation in HM3D (Habitat).

**Focus pivot (Mar 2026 meeting):** shifted from VLA + UNITE injection to world model performance with 3D vs 2D scene understanding.

## Supervisors

- **Prof. Dr. Daniel Braun** (Erstprüfer): Head of Neuroinformatics, Uni Ulm; primary supervisor
- **Prof. Dr. Timo Ropinski** (Zweitprüfer): Head of Visual Computing Group, Uni Ulm; expertise in Computer Vision
- **Fabian** (PhD Betreuer): PhD student, direct day-to-day supervisor; contact via Mattermost or email

## Repo Structure

```
docs/           # GitHub Pages — slides + images
  index.html    # Presentation slides
  images/       # Figures referenced by slides
src/            # Source code
tests/          # Tests
```

## Research

**Core question:** Do world models benefit from 3D semantic scene representations over 2D?

- UNITE (Koch et al. 2025): dense 3D semantic features (CLIP + instance + articulation)
- Baseline: DreamerV3 world model with standard 2D observations
- Experiment: compare DreamerV3 w/ UNITE 3D features vs 2D-only on HM3D ObjectNav
- Related: WMNav (IROS'25) — VLM + world model for ObjectNav; DreamerNav — DreamerV3 for indoor nav

## GitHub Pages

Slides: `https://<username>.github.io/Master-Thesis-3D-VLA/`

Current `docs/index.html` (13 slides, Feb 25 2026 supervisor meeting): thesis pivot proposal — UNITE overview, VGGT foundation, research gap, 3 possible approaches (VLA token injection, PointVLA-style, VGGT+semantic DPT fusion), Habitat ObjectNav & ALFRED benchmarks, VLA baseline, open questions, 6-month timeline.

## Notebooks

When modifying or creating notebooks, execute them in-place so outputs are saved:

```bash
uv run jupyter nbconvert --to notebook --execute <notebook>.ipynb --inplace
```

## GPU Execution

This project runs on BWUniCluster (SLURM). GPU access requires `srun` — never run GPU code directly.

**GPU tests:**
```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00 uv run pytest tests/<file> -x -q -k "<test>"
```

**General GPU commands:**
```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 <command>
```

**When to use `srun`:** if the code imports `jax`, `habitat_sim`, `torch.cuda`, or uses `@pytest.mark.gpu`, it needs GPU → wrap with `srun`.

**Partitions:**
| Partition | Use Case | Max Time |
|-----------|----------|----------|
| `dev_gpu_h100` | Testing, validation, quick experiments | 30 min |
| `gpu_h100` | Standard GPU jobs, training | 48h |
