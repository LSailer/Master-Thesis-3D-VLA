# Reporter Templates

Loaded by `/reporter` in Phase 3 (wiki page) and Phase 4 (HTML slides). Not part of the runtime checklist — consult these when actually writing deliverables.

## Wiki Results section

Append to the existing `docs/wiki/experiments/<name>.md` page (after the Configuration section):

```markdown
**Slides**: [<name>.html](../../<name>.html)

## Results

Key metrics, embedded plots via ![](../../output/figures/<name>/...).

## Findings

What we learned. What surprised us. What this means for the research question:
> Do world models benefit from 3D semantic scene representations over 2D?

## Next

What to try next based on these results.
```

Also update the metadata block at the top of the page:
- `**Status**: implemented` → `**Status**: completed`
- Add `**Slides**: <name>.html` line

## HTML slide style

`docs/index.html` is the canonical style reference — match it. Key elements:

- **Background**: dark (read the exact hex from `index.html`, do not invent one)
- **Font**: Inter
- **Accent colors**: pull from the existing palette in `index.html` — do not introduce new colors
- **Animations**: fade-in for slide transitions, matching the existing `transition` declarations
- **Plots**: embedded as `<img src="../output/figures/<name>/...">`, relative paths
- **Self-contained**: single HTML file, no external CSS or JS dependencies

Each experiment slide follows the same shape:

> **Hypothesis** → **Setup** → **Result** (with plot) → **Implication**

This four-beat structure makes the slide deck readable as both a presentation and a static document.

## Synthesis slide ("What We've Learned")

Last slide in the index narrative deck. Captures the cross-experiment chain of reasoning:

- One arrow per completed experiment (`E1 raised Q1 → E2 tested Q1, found X → E3 …`)
- Ends with the current open question (which becomes the next experiment's hypothesis)
- Link to the most recent experiment deck for readers who want detail

Replace the previous synthesis slide on every `/reporter` run — the chain grows with each experiment.
