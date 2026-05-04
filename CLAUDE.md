# CLAUDE.md

## Project

**World Models + 3D Scene Understanding** — Master's thesis testing whether R2Dreamer performs better with 3D (VGGT) vs 2D features on HM3D ObjectNav.

**Supervisors:**
- Prof. Dr. Daniel Braun (Erstprüfer) — Head of Neuroinformatics, Uni Ulm
- Prof. Dr. Timo Ropinski (Zweitprüfer) — Head of Visual Computing Group, Uni Ulm
- Fabian (PhD Betreuer) — day-to-day supervisor (Mattermost / email)

## Knowledge

Start at [`docs/wiki/index.md`](docs/wiki/index.md). Two-tier: `output/` is raw artifacts (immutable, machine-emitted); `docs/wiki/` is curated synthesis. Never hand-edit anything under `output/`.

Prior session learnings live in [`docs/wiki/lessons/`](docs/wiki/lessons/) — skim today's file before starting related work. Wiki conventions (frontmatter, index/log discipline) live in [`docs/wiki/CLAUDE.md`](docs/wiki/CLAUDE.md) and load when working in that subtree.

## Data layout

- `output/runs/<family>/<slug>-<jobid>/` — per-run dirs. Each holds an auto-emitted `MANIFEST.json` (git_sha, config, wandb_id, slurm_id, timestamps) — never hand-edited. `_blessed/<alias>` symlinks stabilise wiki references against reruns.
- `output/methods/` — non-run artifacts (parity, profiling, comparisons, scenes).
- `output/slurm/` — SLURM logs without run-context.

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Pure canonical vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
