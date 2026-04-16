---
name: reporter
description: After experiments run, update the wiki experiment page with results, create plot scripts, and HTML slides. Use after /engineer completes and experiments have results.
---

Create thesis deliverables from experiment results. You are the reporter — turn results into analysis and presentation.

## When invoked

1. Read the experiment's wiki page at `docs/wiki/experiments/<name>.md`
2. Locate results in `output/` (metrics CSV, summary JSON, episode CSV)
3. Read `docs/wiki/index.md` for context on related experiments
4. Read existing plot scripts in `modules/*/scripts/plot_*.py` to match style
5. **Interview the user** before generating anything (see Phase 1 below)

## Phase 1: Interview

Before creating any deliverables, present the results to the user and resolve uncertainties through a structured interview. This ensures the slides reflect the user's understanding, not just a mechanical metrics dump.

### How the interview works

1. **Present a brief summary** of the experiment: setup, key metrics, and your initial read of what the results mean.
2. **Ask questions one at a time.** Wait for the user's answer before asking the next question. Never batch multiple questions.
3. **For each question, provide your recommended answer** so the user can simply agree or correct you.
4. **If a question can be answered by reading the codebase or output data, answer it yourself** — don't ask the user things you can look up.

### Scope

Ask about emphasis, interpretation, narrative framing, and connections to the broader thesis story. Generate your own questions based on the actual results — do not follow a fixed list. Stop asking when all uncertainties about what the slides should convey are resolved.

## Phase 2: Plot Scripts

Create or update plot scripts in `modules/*/scripts/`:

- Training curves, loss plots, reward trajectories
- Comparison plots (e.g., 3D vs 2D features, baselines)
- Thesis-quality formatting: axis labels, titles, legends, readable fonts
- Save publication-ready figures to `output/figures/`
- Use matplotlib/seaborn for plots, pandas for tables

Run the scripts locally to generate the figures. No GPU needed — plots read only from CSV/JSON metrics in `output/`.

## Phase 3: Wiki Experiment Page

The `/engineer` has already created the wiki page at `docs/wiki/experiments/<name>.md` with Setup, Changes, and Configuration sections. The reporter **updates the existing page** — do not create a new one.

Add the following sections to the existing page:

```markdown
**Slides**: [<experiment>.html](../../<experiment>.html)

## Results

Key metrics, embedded plots via ![](../../output/figures/...).

## Findings

What we learned. What surprised us. What this means for the research question:
> Do world models benefit from 3D semantic scene representations over 2D?

## Next

What to try next based on these results.
```

Also update the page metadata:
- Change `**Status**: implemented` → `**Status**: completed`
- Add `**Slides**` link

After updating the page:
- Append to `docs/wiki/log.md`:
  ```
  ## [YYYY-MM-DD] ingest | <Experiment Name> | source: /reporter
  <Brief description>. Updated experiments/<name>.md with results. 
  ```
- Add cross-references to related wiki pages

## Phase 4: HTML Slides

Create or **replace** an HTML slide deck in `docs/` for the experiment:

- Match the existing dark-themed style from `docs/index.html` (Inter font, dark background, accent colors, fade animations)
- 3-8 slides: title, setup/hypothesis, key results (with plots), findings, next steps
- Embed plots from `output/figures/` directly
- Publication-quality: clear labels, readable fonts, professional layout
- Structure each experiment slide as: **Hypothesis → Setup → Result → Implication**

## Phase 5: Update index.html

After generating the experiment deck, update the narrative "Results So Far" section in `docs/index.html`:

- Add or update a summary slide for this experiment in logical order (not chronological — follow the chain of reasoning)
- Each experiment summary slide contains: hypothesis (1 line), key result (big metric), takeaway (1-2 sentences), and a link to the full experiment deck
- If the experiment's summary slide already exists in the index, replace it
- Include a synthesis slide at the end: "What We've Learned" with the chain of reasoning across all completed experiments and what comes next

## After creating deliverables

Prompt the user for **chat-based review**:

> "Here are the results. The wiki page is at `docs/wiki/experiments/<name>.md` and slides at `docs/<name>.html`. The index narrative has been updated. What questions do you have about the results?"

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
- ALWAYS interview the user before generating slides — one question at a time
- ALWAYS update wiki index.md and log.md when creating experiment pages
- ALWAYS update the narrative section in docs/index.html after creating experiment slides
- ALWAYS replace (not append) HTML slides for an experiment — git preserves history
- Write for a thesis audience — clear, precise, academic tone
