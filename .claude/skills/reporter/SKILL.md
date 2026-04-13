---
name: reporter
description: After review approval, create wiki experiment page, plot scripts, and HTML slides from results. Then prompt user for chat-based review. Use after /review completes and experiments have run.
---

Create thesis deliverables from experiment results. You are the reporter — turn results into analysis and presentation.

## When invoked

1. Check if you're on a GPU node: run `nvidia-smi`. If it succeeds, run commands directly. If it fails, prefix GPU commands with `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00`
2. Review what was built (from the conversation or review summary)
3. Locate results in `output/` and code in `modules/`
4. Read `docs/wiki/index.md` for context on related experiments
5. Create deliverables in three categories

## 1. Plot Scripts

Create or update plot scripts in `modules/*/scripts/`:

- Training curves, loss plots, reward trajectories
- Comparison plots (e.g., 3D vs 2D features, baselines)
- Thesis-quality formatting: axis labels, titles, legends, readable fonts
- Save publication-ready figures to `output/figures/`
- Use matplotlib/seaborn for plots, pandas for tables

Run the scripts to generate the figures.

## 2. Wiki Experiment Page

Create a page in `docs/wiki/experiments/` using the experiment template:

```markdown
# <Experiment Name>

**Status**: completed
**Date**: YYYY-MM-DD
**Tags**: #relevant #tags
**Wandb**: <run URL if available>
**SLURM Job ID**: <job id if available>
**Slides**: [<experiment>.html](../../<experiment>.html)

## Setup

What was tested and why. Hypothesis in one sentence.

## Changes

What changed compared to the previous run.

## Results

Key metrics, embedded plots via ![](../../output/figures/...).

## Findings

What we learned. What surprised us. What this means for the research question:
> Do world models benefit from 3D semantic scene representations over 2D?

## Next

What to try next based on these results.
```

After writing the page:
- Update `docs/wiki/index.md` — add entry under Experiments
- Append to `docs/wiki/log.md`:
  ```
  ## [YYYY-MM-DD] ingest | <Experiment Name> | source: /reporter
  <Brief description>. Created experiments/<name>.md. Updated index.
  ```
- Add cross-references to related wiki pages

## 3. HTML Slides

Create or **replace** an HTML slide deck in `docs/` for the experiment:

- Match the existing dark-themed style from `docs/index.html`
- 3-8 slides: title, setup/hypothesis, key results (with plots), findings, next steps
- Embed plots from `output/figures/` directly
- Publication-quality: clear labels, readable fonts, professional layout

## After creating deliverables

Prompt the user for **chat-based review**:

> "Here are the results. The wiki page is at `docs/wiki/experiments/<name>.md` and slides at `docs/<name>.html`. What questions do you have about the results?"

Answer questions from wiki knowledge and output data. If a question can't be resolved, offer to create a GitHub issue:

```bash
gh issue create --label question --title "<question>" --body "..."
```

## Conventions

- Scripts compute & save to `output/`, wiki pages and slides only reference saved outputs
- One experiment per wiki page
- Always compare against baselines when available
- Frame all findings in terms of the thesis research question

## Rules

- NEVER recompute results — only visualize and analyze what's in `output/`
- ALWAYS update wiki index.md and log.md when creating experiment pages
- ALWAYS replace (not append) HTML slides for an experiment — git preserves history
- Write for a thesis audience — clear, precise, academic tone
