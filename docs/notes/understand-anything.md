# Understand-Anything — using it on the 3D-VLA thesis codebase

> Tool: <https://github.com/Lum1104/Understand-Anything> — a Claude Code plugin that turns
> a codebase into an interactive **knowledge graph** ("Map") you can explore, search,
> diff, and ask questions about. Hybrid engine: tree-sitter for deterministic parsing
> (imports/defs/calls) + LLM agents for semantic summaries and architecture layers.

## TL;DR for this repo

You are already in Claude Code, so install it as a plugin and point it at the repo:

```text
/plugin marketplace add Lum1104/Understand-Anything
/plugin install understand-anything
/understand                 # builds .understand-anything/knowledge-graph.json
/understand-dashboard       # opens the interactive Map in a browser
```

It supports Python, so it will parse `agents/`, `visualizations/`, `conftest.py`, etc.

---

## 1. Building & using the "Map" (knowledge graph)

The "Map" is the interactive knowledge graph. Every file / function / class becomes a
clickable node; edges are imports and call relationships.

```text
/understand                 # scan repo, extract files+functions+classes+deps, build graph
/understand-dashboard       # open the visual graph in the browser
/understand-chat            # ask questions in plain English ("where is the reward computed?")
/understand-explain <path>  # deep-dive a single file/module
/understand-domain          # extract business/domain flows
/understand-onboard         # generate a team/onboarding guide
```

What the Map gives you:
- **Search** by name (fuzzy) or by meaning ("which parts handle the WP/CP grid?").
- **Layer view** — auto-groups nodes into architectural layers (API/Service/Data/UI/Util).
- **Guided tours** ordered by dependency, so you read the code in a sensible order.
- **Plain-English summaries** of what each node does and how nodes relate.

The graph is saved to `.understand-anything/knowledge-graph.json` (plain JSON). Commit it
once and collaborators skip regeneration. Use git-lfs if it gets large. Add the
post-commit hook to keep it fresh:

```text
/understand --auto-update    # incremental graph updates on each commit
```

> Note for this repo: add `.understand-anything/` to `.gitignore` unless you deliberately
> want to version the graph. The thesis repo already tracks generated HTML under `docs/`,
> so decide consciously whether the graph belongs in version control.

## 2. Checking the changes you made

Use the diff/impact command. It shows which parts of the system your *uncommitted* changes
touch and the ripple effects, **before** you commit.

```text
/understand-diff            # impact analysis of current working-tree changes
```

This is the "what did my change affect" view — complementary to `git diff` (which shows
*lines*) because it shows *which graph nodes/components* are affected downstream.

## 3. Doing a code review

There is no single "review" command; you compose three:

| Step | Command | Purpose |
|------|---------|---------|
| 1. Scope the blast radius | `/understand-diff` | see which modules/nodes the change hits |
| 2. Drill into a module | `/understand-explain <path>` | understand the file being changed in depth |
| 3. Reason about architecture | `/understand-chat` | ask "does this break the X→Y flow?" |

Add `--review` for full LLM validation of the generated graph/analysis:

```text
/understand --review
```

Important: this is **comprehension-oriented** review (impact + architecture understanding),
not a line-by-line bug/security linter. For correctness bugs and security issues on your
branch, keep using Claude Code's own `/code-review` and `/security-review`.

## 4. Install options (reference)

```text
# Native Claude Code (recommended here)
/plugin marketplace add Lum1104/Understand-Anything
/plugin install understand-anything

# One-line installer (macOS/Linux) — also wires up Codex/Cursor/Copilot/Gemini CLI
curl -fsSL https://raw.githubusercontent.com/Lum1104/Understand-Anything/main/install.sh | bash
```

Useful flags: `--language zh|ja|ko|ru` (localized output), `--auto-update` (commit hook),
`--review` (full LLM validation).

## 4b. Where to run it — run LOCALLY, not on the HPC (chosen approach)

The HPC login node (`ul_hfj15@...`) is **headless**: `/understand-dashboard` serves/open a
browser to render the Map, and there is no display on the cluster. You'd also fight pnpm
native builds (pnpm 11 blocks tree-sitter/esbuild build scripts by default; `ERR_PNPM_IGNORED_BUILDS`).
Because the graph is a portable JSON file, build *and* view it on your laptop instead.

**Local playbook (run on your own machine, where you have a browser):**

```bash
# 1. Shallow-clone just the working branch (skip the heavy external/ wandb/ output/ dirs)
git clone --depth 1 --branch lucasailerls/3d-50-hybrid-cnn-vggt <repo-url> 3d-vla
cd 3d-vla
# optional: keep it lean with sparse-checkout (src is the real code)
#   git sparse-checkout init --cone && git sparse-checkout set src tests docs

# 2. Node >= 22 and pnpm >= 10 (locally easy via corepack)
corepack enable && corepack prepare pnpm@latest --activate

# 3. In Claude Code (local): install the plugin, then RESTART the session
#    /plugin marketplace add Lum1104/Understand-Anything
#    /plugin install understand-anything
# 4. Build the graph scoped to src, then open the Map
#    /understand src
#    /understand-dashboard
```

Note: freshly installed plugin skills only activate in a **new** Claude Code session.
First `/understand` run builds `@understand-anything/core` once (needs the build scripts
approved — pnpm will prompt `pnpm approve-builds`, or add an `onlyBuiltDependencies` list
to `pnpm-workspace.yaml`).

**Hybrid alternative (only if you want HPC compute to do the analysis):** build the graph on
the HPC, then `git add -f .understand-anything/knowledge-graph.json`, pull it locally, and run
`/understand-dashboard` locally to view. Skipped here because the HPC pnpm build is the pain
point we're avoiding.

## 5. Alternatives & when to prefer each

| Tool | Type | Strength | Weakness vs Understand-Anything |
|------|------|----------|----------------------------------|
| **Understand-Anything** | LLM + tree-sitter, in Claude Code | Plain-English Map + diff impact + chat, no extra app | Newer; LLM summaries cost tokens; not a bug linter |
| **Sourcetrail** (archived) | Static interactive graph | Mature dep/call graph, offline | Unmaintained; no LLM Q&A; C++/Java/Python focus |
| **Scientific/SciTools "Understand"** | Commercial static analyzer | Deep metrics, dependency, CFGs | Paid license; no LLM chat; heavier setup |
| **CodeSee** | Hosted code maps | Auto diagrams in CI/PRs | SaaS, sign-up; less Python-deep; closing/uncertain |
| **Doxygen + Graphviz** | Doc/graph generator | Free, reproducible call graphs | No semantics/Q&A; verbose config |
| **GitHub Copilot / `/code-review`** | LLM in editor/CLI | Bug & security review, inline | No persistent visual graph of whole repo |
| **Plain `git diff` + grep + ctags** | Built-in | Zero deps, exact | No semantic map, no impact graph |

Rule of thumb for the thesis:
- Want a **visual mental model + ask-questions** of a large unfamiliar area → Understand-Anything.
- Want **bug/security correctness** on a branch → Claude Code `/code-review`, `/security-review`.
- Want **exact line changes** → `git diff`. They stack; they don't replace each other.

## 6. Benefits & trade-offs (for *this* codebase)

Benefits:
- The repo is research code with many interacting pieces (agents, hybrid CNN/VGGT encoder,
  WP/CP grids, SLURM runs). A semantic Map + `/understand-chat` shortens "where does X live"
  questions for you and for your advisor/readers.
- `/understand-diff` is a nice pre-commit sanity check that an encoder change doesn't quietly
  ripple into eval/visualization paths.
- Runs inside Claude Code — no separate server, the graph is just a JSON file.

Trade-offs / cautions:
- LLM summaries can be wrong or stale; treat the Map as a guide, not ground truth — verify in code.
- Building/refreshing the graph spends tokens (parallel agents, 20–30 files/batch).
- It does **not** find bugs, race conditions, or numerical issues — pair it with real review.
- Decide whether to commit `.understand-anything/` (reproducibility vs repo bloat).

## Sources
- <https://github.com/Lum1104/Understand-Anything>
- <https://github.com/Lum1104/Understand-Anything/blob/main/README.md>
- <https://understand-anything.com/>
- <https://dev.to/arshtechpro/understand-anything-turn-any-codebase-into-an-interactive-knowledge-graph-37ed>
