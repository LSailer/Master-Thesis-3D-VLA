---
name: reporter
description: Turn finished experiment results into thesis deliverables — plot scripts, an updated wiki experiment page, an HTML slide deck, and an updated narrative on the docs index. Use after /engineer completes and experiments have metrics in output/, or when the user wants to refresh the slides for an existing experiment.
---

# Reporter

## Quick start

> **User:** `/reporter l2-vggt-baseline`
> **You:** "Reading `docs/wiki/experiments/l2-vggt-baseline.md` and `output/runs/l2-vggt-baseline/`. Headline: SR=0.42 vs CNN baseline 0.36. Hypothesis (world models benefit from 3D) confirmed at L2. Two questions before I draft slides: (1) lead with the SR delta or with the SPL delta? (2) compare against L1 (continuity story) or only L2 (clean baseline)?"

Wiki Results-section template and HTML slide style live in [TEMPLATES.md](TEMPLATES.md). For autonomous reporting (auto-pipeline Phase 4) the codepath is `scripts/pipeline/report_prompt.md` — a separate prompt, not this skill.

## Workflow

### Setup

- [ ] Read `docs/wiki/experiments/<name>.md` (Setup, Changes, Configuration)
- [ ] Locate results in `output/` (metrics CSV, summary JSON, episode CSV)
- [ ] Skim `docs/wiki/index.md` for related experiments
- [ ] Read existing `modules/*/scripts/plot_*.py` for style conventions

### Phase 1 — Interview (one question at a time)

Run a brief grill-me-style interview about emphasis and narrative — never batch. Cover only what shapes the slides:

- [ ] Lead metric and framing
- [ ] Comparison set (which baselines to show)
- [ ] Connection to thesis question (3D > 2D for world models)
- [ ] What was surprising vs expected

Skip questions whose answers are obvious from the data — answer those yourself.

### Phase 2 — Plot scripts

- [ ] Create or update plot scripts in `modules/*/scripts/plot_*.py` (matplotlib/seaborn, pandas for tables)
- [ ] Save publication-quality figures to `output/figures/<name>/...`
- [ ] Run the scripts locally — no GPU needed; plots read CSV/JSON only

### Phase 3 — Wiki experiment page

The page already exists from `/engineer`. Update in place — do not create a new one.

- [ ] Append Results, Findings, Next sections per the template in [TEMPLATES.md](TEMPLATES.md)
- [ ] Update frontmatter `**Status**: implemented` → `**Status**: completed`
- [ ] Add `**Slides**` link pointing to `docs/<name>.html`
- [ ] Ensure the page has YAML frontmatter at the very top (above the H1) with the new layout contract from `docs/wiki/recaps/2026-04-26-output-restructure.md` (decisions #7 and #10):
  ```yaml
  ---
  run_path: output/runs/<family>/_blessed/<alias>
  slurm_id: <id>
  wandb_id: <id>
  status: blessed
  ---
  ```
  `<family>` is the run-family directory under `output/runs/` (e.g. `r2dreamer-curriculum-l2-vggt`); `<alias>` is the human slug (typically the `<name>` argument). If the page already has frontmatter, preserve other fields and add the new ones.
- [ ] Create the `_blessed/<alias>` symlink in the runs dir pointing at the actual job-id-suffixed run dir, so wiki refs survive reruns:
  ```bash
  mkdir -p output/runs/<family>/_blessed
  ln -snf ../<slug>-<jobid> output/runs/<family>/_blessed/<alias>
  ```
  Use a relative target (`../<slug>-<jobid>`) so the symlink survives directory moves. If the alias already exists pointing somewhere else, ask the user before re-pointing.
- [ ] Append a one-liner to `docs/wiki/log.md`:
  ```
  ## [YYYY-MM-DD] ingest | <Experiment Name> | source: /reporter
  <one-paragraph result summary>. Updated experiments/<name>.md with results.
  ```
- [ ] Add cross-references to related experiment pages

### Phase 4 — HTML slides

- [ ] Create or replace `docs/<name>.html` (replace, not append — git preserves history)
- [ ] Match the dark-themed style from `docs/index.html` per [TEMPLATES.md](TEMPLATES.md)
- [ ] 3–8 slides: title, setup+hypothesis, key results with plots, findings, next steps
- [ ] Each experiment slide: Hypothesis → Setup → Result → Implication
- [ ] Embed plots from `output/figures/<name>/` via relative paths

### Phase 5 — Update narrative index

- [ ] Update or add the experiment's summary slide in `docs/index.html` "Results So Far" — logical order (chain of reasoning), not chronological
- [ ] Summary slide: hypothesis (1 line), key result (big metric), takeaway (1–2 sentences), link to full deck
- [ ] Replace if already present
- [ ] End with a "What We've Learned" synthesis slide spanning all completed experiments

### After deliverables

Prompt the user for chat-based review: "Wiki at `docs/wiki/experiments/<name>.md`, slides at `docs/<name>.html`, index narrative updated. Questions?" Answer from wiki + output. For unresolved questions, offer `gh issue create --label question --title "<q>" --body "..."`.

## Rules

- Never recompute — only visualise what's already in `output/`
- One experiment per wiki page; always compare against baselines when available
- Frame findings against the thesis research question (3D > 2D world models)
- Thesis-audience tone: clear, precise, academic
