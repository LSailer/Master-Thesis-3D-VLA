---
name: wiki-audit
description: Audit docs/wiki/ pages against external artifacts in output/. Three operations — metrics (cross-check SR/SPL/% claims vs metrics.csv + wandb ids in log.md), figures (orphan figures vs unbacked plot claims), all (runs both). Use after a training run finishes, before writing slides, or before /defense rehearsal. Complementary to /wiki lint, which is wiki-internal.
---

# Wiki Audit

Cross-check claims in `docs/wiki/` against the artifacts in `output/` that are supposed to back them up. `/wiki lint` checks the wiki against itself; this skill checks it against the training results.

All reports write to `output/audits/<op>-<YYYYMMDD-HHMM>.md` (gitignored). Print a one-line summary to stdout with the report path.

## Operations

### Invocation

- `/wiki-audit metrics` — cross-check numeric claims vs artifacts.
- `/wiki-audit figures` — cross-check figure references vs `output/figures/` + per-experiment `output/` plots.
- `/wiki-audit all` — runs both, two separate reports.
- `/wiki-audit` (no arg) — infer from context: if the most recent 5 commits touched `output/**/*.csv` or `output/**/*.json`, run `metrics`; if they touched `output/figures/` or added `![](...)` to a wiki page, run `figures`; if both or neither, ask the user which.

Free-text context is accepted as a hint, e.g. `/wiki-audit "focus on L2"`. Use it to narrow the scan but never to hard-filter — still list everything; just rank the hinted area first.

### metrics

**What it checks.** Every numeric claim in `docs/wiki/experiments/*.md` and `docs/wiki/methods/*.md` against its source of truth.

**Process.**
1. For each page under `docs/wiki/experiments/` and `docs/wiki/methods/`:
   - Parse YAML frontmatter (per `docs/wiki/recaps/2026-04-26-output-restructure.md` decision #10). Required keys: `run_path`, `slurm_id`, `wandb_id`, `status`. If frontmatter is missing or `run_path` is absent, classify the entire page as **unverifiable** with reason `"no run_path frontmatter — see recap 2026-04-26-output-restructure"` and continue.
   - Grep for claims: `\d+(\.\d+)?\s*%`, `\d+(\.\d+)?\s*(SR|SPL)`, "X episodes", "Xx above random", "Xpp over", "p50/p95 = X ms".
2. For each claim, locate the ground-truth artifact via frontmatter:
   - Resolve `run_path` (follow symlinks — `_blessed/<alias>` typically points at `../<slug>-<jobid>/`). If the symlink target does not exist, classify as **unverifiable** with reason `"run_path <path> does not resolve"`.
   - Validate `MANIFEST.json` at `<resolved>/MANIFEST.json`: confirm its `slurm_id` and `wandb_id` match the page frontmatter. Mismatches are **drift** (provenance drift, not value drift) and reported separately.
   - Parse `<resolved>/metrics.csv` / summary `.json` / any `episodes.csv` in that dir for the numeric claims.
3. Classify each claim:
   - **match** — within 0.5pp (percentages) or ±1% (other numerics).
   - **drift** — outside tolerance; report both values + file:line + artifact path.
   - **mid-flight** — artifact mtime < 10 minutes ago OR SLURM job id still in `squeue` OR `metrics.csv` has fewer than 100 rows OR `MANIFEST.json` has no `end_timestamp` yet. Skip; mark as "run not done yet, re-run audit after completion".
   - **unverifiable** — frontmatter missing, `run_path` does not resolve, or no matching column in `metrics.csv`; flag for manual check.
4. Emit report to `output/audits/metrics-<timestamp>.md`. Include a "Provenance drift" section listing pages whose frontmatter `slurm_id`/`wandb_id` disagrees with `MANIFEST.json`.

**Report shape.**

```markdown
# Metrics Audit — <timestamp>

**Scanned:** N pages, M claims
**Match:** X · **Drift:** Y · **Mid-flight:** Z · **Unverifiable:** W

## ⚠ Drift (Y)
| Page | Claim | Said | Actual | Artifact |
|---|---|---|---|---|
| experiments/l1-rerun-buffix.md:12 | "75% SR" | 75.0 | 73.4 | output/runs/r2dreamer-curriculum-l1-rerun/_blessed/l1-rerun-buffix/metrics.csv |

## 🔗 Provenance drift (P)
| Page | Field | Frontmatter | MANIFEST.json |
|---|---|---|---|
| experiments/l1-rerun-buffix.md | wandb_id | y5a0upzd | abc12345 |

## ⏳ Mid-flight (Z)
<list — page, claim, reason>

## ❓ Unverifiable (W)
<list — page, claim, why no artifact match>

## ✓ Matches (X)
<collapsed summary only — one line per page>
```

### figures

**What it checks.** Figure files vs figure references.

**Process.**
1. Enumerate all figure files: `output/figures/*.{png,pdf,svg,tex,gif}` and `output/**/plots/*.{png,pdf}`.
2. Grep `docs/wiki/**/*.md` for `![...](...)`, `\includegraphics{...}`, and bare filename mentions.
3. Classify:
   - **referenced** — figure exists and is cited somewhere.
   - **orphan** — figure exists but no wiki page cites it.
   - **stale ref** — wiki cites a path that no longer exists on disk.
   - **unbacked claim** — wiki sentences with numeric claims (`\d+%`, `\d+\s*ms`) in paragraphs with no nearby figure reference (< 10 lines). Soft signal; don't be strict.
4. Emit to `output/audits/figures-<timestamp>.md`.

### all

Run `metrics` then `figures`. Two separate reports. Print both paths to stdout.

## Fail-soft rules

- If `output/` is missing or empty: print `"output/ is empty — nothing to audit against. Have any training runs finished?"` and exit 0.
- If a wiki page can't be mapped to any `output/` subdir: do not crash; add to `unverifiable`.
- If `git status` shows an uncommitted `output/**/metrics.csv`, trust the file but note in the report footer: `"Note: <path> is uncommitted locally."`

## See also

- Run `/wiki-audit metrics` before `/defense` — question quality depends on claim accuracy.
- Run `/wiki-audit figures` before `/reporter` slide generation.
- `/wiki lint` for wiki-internal consistency (orphan pages, contradictions, stale claims).

## Anti-patterns

- Do not edit any wiki page. Reports only.
- Do not create GitHub issues. The user reviews and triages.
- Do not hard-fail on mid-flight runs — that's a normal state, not an error.
