# Wiki conventions

This wiki is curated synthesis backed by raw artifacts in `output/`. Never duplicate `output/` contents here — link to them.

## Layout

- `index.md` — entry point. Add a link here whenever you create a new page.
- `log.md` — chronological activity log. Append a line whenever you create or substantively edit a page.
- `experiments/` — per-run results. One page per experiment family.
- `methods/` — architecture notes, parity reports, profiling, debugging recipes.
- `meetings/` — supervisor notes, dated.
- `research/` — paper summaries.
- `lessons/` — `YYYY-MM-DD.md` files written by the SessionEnd hook. Read recent ones before starting related work.
- `recaps/` — distillates from human-led design sessions.

## Experiment page frontmatter

Every page in `experiments/` declares its backing run:

```yaml
---
run_path: output/runs/<family>/<slug>-<jobid>/
slurm_id: <id>
wandb_id: <id>
status: running | done | failed | blessed
---
```

When citing numbers (SR, SPL, %), cross-check against `<run_path>/metrics.csv` and `MANIFEST.json` before publishing — claims must be reproducible from the artifact.

## Lessons file format

`lessons/YYYY-MM-DD.md` is appended by the SessionEnd hook with bullets like:

```
## HH:MM - session <short-id>
- [gotcha] one-line lesson — context
- [finding] ...
- [decision] ...
- [deadend] ...
```

Lessons are append-only. If a lesson becomes wrong, add a corrective lesson rather than editing the past.
