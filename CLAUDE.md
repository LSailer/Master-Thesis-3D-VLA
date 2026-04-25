# CLAUDE.md

## Project

**World Models + 3D Scene Understanding** — Master's thesis testing whether DreamerV3 performs better with 3D (VGGT/UNITE) vs 2D features on HM3D ObjectNav. Pivoted Mar 2026 away from VLA + UNITE injection.

**Supervisors:**
- Prof. Dr. Daniel Braun (Erstprüfer) — Head of Neuroinformatics, Uni Ulm
- Prof. Dr. Timo Ropinski (Zweitprüfer) — Head of Visual Computing Group, Uni Ulm
- Fabian (PhD Betreuer) — direct day-to-day supervisor (Mattermost / email)

## Knowledge base

Project knowledge lives in `docs/wiki/`. Start at [`docs/wiki/index.md`](docs/wiki/index.md):

- `experiments/` — training-run results
- `methods/` — architecture, recipes (e.g. [phase-orchestration](docs/wiki/methods/phase-orchestration.md), [launcher-refactor](docs/wiki/methods/launcher-refactor.md), [vggt-r2dreamer-callchain](docs/wiki/methods/vggt-r2dreamer-callchain.md))
- `meetings/` — supervisor notes
- `research/` — paper summaries

When creating or modifying wiki pages, also update `docs/wiki/index.md` and append to `docs/wiki/log.md`. Never modify raw data in `output/`.

## Coding principles

- **Clarify before coding** — present interpretations, ask, don't guess silently.
- **Goal-driven** — define testable success criteria before implementing.
- **Simplicity first** — minimum viable code. Three similar lines beat a clever helper.
- **Surgical changes** — only modify what's necessary, no "while you're in there" cleanups.
- **One question at a time** — wait for answer before next.
