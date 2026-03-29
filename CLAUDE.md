# CLAUDE.md

## Project Overview

Master's thesis: **World Models + 3D Scene Understanding** — testing whether world models (e.g. DreamerV3) perform better with 3D semantic features (UNITE) vs 2D-only for object navigation in HM3D (Habitat).

**Focus pivot (Mar 2026 meeting):** shifted from VLA + UNITE injection to world model performance with 3D vs 2D scene understanding.

## Supervisors

- **Prof. Dr. Daniel Braun** (Erstprüfer): Head of Neuroinformatics, Uni Ulm; primary supervisor
- **Prof. Dr. Timo Ropinski** (Zweitprüfer): Head of Visual Computing Group, Uni Ulm; expertise in Computer Vision
- **Fabian** (PhD Betreuer): PhD student, direct day-to-day supervisor; contact via Mattermost or email

## Coding Principles

- **Clarify before coding**: When requirements are ambiguous, present interpretations and ask — don't guess silently.
- **Goal-driven**: Define testable success criteria before implementing. State what "done" looks like.
- **Simplicity first**: Write the minimum viable code. No premature abstractions, no speculative error handling, no "just in case" features. Three similar lines are better than a clever helper nobody asked for.
- **Surgical changes**: Only modify what's necessary. Don't clean up unrelated code, add unrequested features, or refactor "while you're in there."
- **One question at a time**: Always ask one question at a time. Never dump multiple questions or recommendations at once. Wait for the answer before proceeding.

## Repo Structure

```
modules/                    # All code organized by domain
  dreamerv3/                # DreamerV3 world model (JAX)
    agent.py, networks.py, configs.py, optim.py, replay_buffer.py, train.py, eval.py
    tests/                  # pytest: test_shapes.py, test_integration.py
    notebooks/              # jax_vs_pytorch, official_comparison
    scripts/                # run_ours_crafter.py, slurm/
  r2dreamer/                # R2-Dreamer agent (extends DreamerV3)
    agent.py, networks.py, config.py
    tests/                  # test_agent.py, test_shapes.py, test_optim.py
    scripts/                # run_jax_crafter.py, run_pytorch_*.py
  envs/                     # Environment wrappers (Crafter, Habitat)
    crafter.py, habitat.py, habitat_r2dreamer.py
    tests/                  # test_habitat.py
    notebooks/              # habitat_headless_test, habitat_objectnav_benchmark
    scripts/                # run_habitat_timing.py, collect_*.py
  vggt/                     # VGGT comparison & benchmarks
    benchmark.py, plots.py, variants.py
    tests/                  # test_comparison.py
    notebooks/              # comparison, exploration
  shared/                   # Cross-cutting utilities
    wandb_utils.py
external/                   # Third-party repos (dreamerv3-official, r2dreamer, VGGT variants)
scripts/                    # Cross-cutting scripts (smoke tests, setup, SLURM)
docs/                       # GitHub Pages — slides + images
archiv/                     # Deprecated code (ALFRED, old notebooks)
output/                     # Run results & benchmarks
data/                       # Datasets (HM3D scenes, etc.)
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

