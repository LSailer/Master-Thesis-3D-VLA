---
date: 2026-04-26
skill: /grill-me
topic: Output directory restructure to LMWiki pattern
status: spec-locked, ready for /engineer-team
reference: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
---

# Output Restructure to LMWiki Pattern

## Context

Karpathy's LLM-Wiki gist proposes a **2-tier architecture**: immutable raw sources + LLM-generated wiki + a CLAUDE.md schema between them. This project actually has **3 tiers** in practice (raw artifacts, computed derivatives, conceptual synthesis), and `output/` currently mixes the first two while loose orphan files float at the root. This spec resolves the mismatch.

## Decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Layer model | **Hybrid**: per-run raw + per-run derived stay together; cross-run derivatives separate | Preserves notebook workflow ([feedback_notebook_workflow](memory)) |
| 2 | Folder for Claude session distillates | **`docs/wiki/recaps/`** with `YYYY-MM-DD-<topic>.md` | Artifact-named, parallel to `meetings/` |
| 3 | `output/` top-level buckets | **`runs/`, `methods/`, `slurm/`** | 3 buckets sufficient |
| 4 | aggregated/ vs methods/ | **Merged** as `methods/comparisons/` | Simplicity over precision; user override |
| 5 | jax-vs-pytorch grouping | Own sub-folder under comparisons: **`methods/comparisons/jax-vs-pytorch/`** | User override |
| 6 | parity location | **`methods/parity/` separate** | Distinct concept (correctness, not comparison) |
| 7 | Run dir naming | **`<slug>-<jobid>/`** + **`_blessed/<alias>` symlinks** for wiki-cited runs | Readable + traceable + stable wiki refs |
| 8 | SLURM logs location | **Inside run dir** as `slurm.out` / `slurm.err` (no parent-level loose files) | Co-located with run |
| 9 | Per-run manifest | **Auto-emitted `MANIFEST.json`** by training script (git-sha, config, wandb-id, slurm-id, timestamps) | Drift-free, immutable |
| 10 | Wiki ↔ raw linkage | **Frontmatter** `run_path` / `slurm_id` / `wandb_id` on every experiment page | Deterministic `wiki-audit` cross-check |
| 11 | Migration strategy | **Big-Bang** (one PR, all changes) | User chose accelerated cleanup over rolling |
| 12 | CLAUDE.md schema | Add new **"Data layout"** section codifying the contract | Make schema explicit for future LLM context |
| 13 | PR scope | All four follow-on items go INTO the Big-Bang PR | Consistent with Big-Bang choice |

## Final Layout

```
output/
├── runs/                                    # Training/eval runs
│   └── r2dreamer-curriculum-l1-vggt/
│       ├── baseline-actent3e-4-3954649/    # <slug>-<jobid>/
│       │   ├── MANIFEST.json                # auto-emitted at run start
│       │   ├── metrics.csv                  # raw
│       │   ├── slurm.out                    # SLURM logs MOVED in
│       │   ├── slurm.err
│       │   └── plots/                       # per-run derived plots
│       ├── baseline-actent3e-4-3957756/    # rerun, sorts adjacent
│       └── _blessed/
│           └── baseline-actent3e-4 -> ../baseline-actent3e-4-3954649
├── methods/
│   ├── comparisons/                         # cross-run + cross-framework
│   │   ├── jax-vs-pytorch/                  # perf-bench + training-quality JAX↔PT
│   │   ├── 5way_episode_score.png
│   │   └── summary_table_5way.csv
│   ├── parity/                              # bit-level JAX↔PT (separate concept)
│   ├── profiling/                           # per-phase timing
│   ├── scenes/                              # dataset/scene characterization
│   │                                        # (floorplan, scene_analysis, goal_distance,
│   │                                        # 3d_pointclouds, shortest_path)
│   └── autoresearch/                        # perf-race outputs
└── slurm/                                   # only jobs without a run-context

docs/wiki/
├── recaps/                                  # NEW: Claude-Code session distillates
├── figures/                                 # NEW: curated slide/defense visuals
│   └── defense/                             # was output/defense/
├── experiments/  methods/  meetings/  research/  _templates/

archiv/
└── debug_session-20260426/                  # was output/debug_session/
```

## PR Plan (Big-Bang)

**Step 0 — Pre-flight (must complete first):**
```bash
grep -rn "output/" --include='*.py' --include='*.ipynb' . > /tmp/output-paths.txt
```
Build a checklist of every hardcoded path. Estimate effort before committing.

**Step 1 — Layout migration (`git mv`):**
- All `r2dreamer-curriculum-*/` → `output/runs/<family>/`
- `output/comparison/*` → `output/methods/comparisons/`
- `output/parity/` → `output/methods/parity/`
- `output/profiling/` → `output/methods/profiling/`
- `output/autoresearch/` → `output/methods/autoresearch/`
- `output/{floorplan*,scene_analysis,goal_distance*,3d_pointclouds,shortest_path*}` → `output/methods/scenes/`
- `output/jax_vs_pytorch_benchmark.*` → `output/methods/comparisons/jax-vs-pytorch/`
- `output/figures/` → inspect; curated → `docs/wiki/figures/`, scripted → `output/methods/comparisons/`
- `output/defense/` → `docs/wiki/figures/defense/`
- `output/debug_session/` → `archiv/debug_session-20260426/`

**Step 2 — Per-run cleanup:**
- Move parent-level `slurm-*.{out,err}` INTO their corresponding `run-<jobid>/` dirs as `slurm.{out,err}`
- Rename `run-<jobid>/` to `<slug>-<jobid>/` where slug is derivable from existing semantic-slug siblings (e.g., `baseline-actent3e-4`); leave others as-is
- Create `_blessed/` symlinks for every run cited in the 6 existing experiment pages

**Step 3 — Code path updates:**
- Every hit from Step 0's checklist: update or grep-replace
- Add MANIFEST.json emission to training-script entry point (writes git-sha, config snapshot, wandb-id, slurm-id from env, start timestamp, end timestamp)

**Step 4 — Wiki frontmatter backfill:**
- All 6 experiment pages get frontmatter:
  ```yaml
  ---
  run_path: output/runs/<family>/_blessed/<alias>
  slurm_id: <id>
  wandb_id: <id>
  status: blessed
  ---
  ```

**Step 5 — Skill updates** (via `/write-a-skill`, NOT manual edit per [feedback_skills_via_write_a_skill](memory)):
- `slurm-submit`: ensure submission script sets env vars consumed by MANIFEST emission
- `reporter`: write frontmatter when creating new experiment pages
- `wiki-audit`: read frontmatter `run_path`, follow symlink, validate claims against `MANIFEST.json` + `metrics.csv`

**Step 6 — Schema update:**
- Add "Data layout" section to `CLAUDE.md` (~15 lines codifying buckets, naming, frontmatter, MANIFEST contract)
- Update `docs/wiki/index.md` with `Recaps` section
- Append entry to `docs/wiki/log.md`
- Add `docs/wiki/_templates/recap.md` template

## Eval Criteria

Big-Bang PR is **done** when all of the following hold:

1. `grep -rn "output/comparison\|output/r2dreamer-\|output/parity\|output/profiling" --include='*.py' --include='*.ipynb' .` returns **zero hits** (all paths migrated)
2. Re-running every notebook in `notebooks/` succeeds without `FileNotFoundError`
3. `/wiki-audit metrics` returns **zero unmatched claims** (every `%`/`SR`/`SPL` number traces to a `metrics.csv` via frontmatter)
4. `ls output/` returns exactly: `runs/`, `methods/`, `slurm/` (no loose files)
5. All 6 experiment pages in `docs/wiki/experiments/` parse with valid frontmatter pointing to a real `runs/` path
6. `CLAUDE.md` contains a "Data layout" section that names the buckets and the frontmatter contract
7. New training run produces a valid `MANIFEST.json` in its run dir

## Out of Scope

- Retrofitting `MANIFEST.json` into pre-existing run dirs — old runs are immutable raw, leave them
- Renaming `output/` itself to `raw/` — too many path-rewrites for marginal gain

## Next Step

Hand this recap to `/engineer-team` for implementation, OR drive it manually starting with Step 0.
