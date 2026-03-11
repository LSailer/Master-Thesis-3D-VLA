# Ralph Pipeline Handbook

Terse reference for the autonomous dev agent pipeline.

## 1. Quick Reference

| Command | What it does |
|---------|-------------|
| `/write-a-prd` | Interactive 5-step PRD creation → GitHub issue with `prd` label |
| `/prd-to-issues #N` | Decompose PRD into vertical-slice task issues with deps/labels |
| `/slurm-submit` | Trigger `ralph.yml` workflow on BWUniCluster via GitHub Actions |
| `/ralph-loop` | Run ralph locally: scan ready tasks → TDD → PR |
| `/qa-review` | Review `in-review` PRs: approve/request changes/reject |
| `/tdd` | Manual TDD mode: red-green-refactor one test at a time |

## 2. Board Flow

```
Backlog → Ready → In Progress → In Review → Done
 (human)  (human)   (ralph)      (human)    (human)
```

**Human gate**: only humans move issues from Backlog → Ready. Ralph never picks from backlog.

### Label ↔ Column Mapping

| Label | Board Column | Who sets it |
|-------|-------------|-------------|
| `backlog` | Backlog | `/prd-to-issues` (default for new tasks) |
| `ready` | Ready | Human (promotes from backlog) |
| `in-progress` | In Progress | Ralph (claims task) |
| `in-review` | In Review | Ralph (after PR creation) |
| `blocked` | — (stays in column) | Ralph (3 consecutive test failures) |
| `AFK` | — (cross-cutting) | `/prd-to-issues` (fully automatable) |
| `HITL` | — (cross-cutting) | `/prd-to-issues` (needs human input) |

Board: [Project #2](https://github.com/orgs/LSailer/projects/2)

## 3. End-to-End Walkthrough: New Feature

1. **`/write-a-prd`** — answer problem exploration + design interview questions → PRD issue created with `prd` label
2. **`/prd-to-issues #N`** — decomposes PRD into task issues, each gets `AFK`/`HITL` + `backlog` labels
3. **Move tasks Backlog → Ready** — on the board, drag tasks you want ralph to work on; this swaps `backlog` → `ready` label
4. **`/slurm-submit`** (cluster) or **`/ralph-loop`** (local) — ralph picks lowest-numbered unblocked `AFK` + `ready` issue, TDD-implements it in a worktree, creates PR, labels `in-review`
5. **`/qa-review`** — review each PR: see diff, check results, then approve (squash merge + close issue), request changes (back to `in-progress`), or reject

## 4. Migrating Existing Issues

### Required for ralph compatibility

- Labels: `AFK` + `ready` (or `backlog` if not promoting yet)
- Body must contain **Acceptance Criteria** as a checklist:
  ```markdown
  ## Acceptance Criteria

  - [ ] criterion 1
  - [ ] criterion 2
  ```
- Body must contain **Blocked By** section (even if empty):
  ```markdown
  ## Blocked By

  None
  ```

### Optional sections

- `## Parent PRD` — reference to PRD issue (`Closes criteria in #N`)
- `## Technical Notes` — compute reqs, key files, interfaces

### Issue body template (copy-paste)

```markdown
## Parent PRD

Closes criteria in #<PRD_NUMBER>

## Description

<what this task implements>

## Acceptance Criteria

- [ ] <testable criterion 1>
- [ ] <testable criterion 2>

## Blocked By

None

## Technical Notes

- Compute: CPU
- Key files: `src/<path>`, `tests/<path>`
```

### Batch migration

```bash
# Add labels to existing issues
gh issue edit 10 11 12 --add-label AFK --add-label backlog

# Promote to ready
gh issue edit 10 --remove-label backlog --add-label ready

# Add to project board
gh project item-add 2 --owner LSailer --url https://github.com/LSailer/Master-Thesis-3D-VLA/issues/10
```

## 5. Label Reference

| Label | Color | Meaning | Who sets | Transitions |
|-------|-------|---------|----------|-------------|
| `prd` | `#0075ca` | Product Requirements Document | `/write-a-prd` | — |
| `AFK` | `#2ea44f` | Fully automatable by ralph | `/prd-to-issues` | — |
| `HITL` | `#e4e669` | Needs human decisions | `/prd-to-issues` | — |
| `backlog` | `#c5def5` | Not yet started | `/prd-to-issues` | → `ready` (human) |
| `ready` | `#2ea44f` | Ready for ralph | Human | → `in-progress` (ralph) |
| `in-progress` | `#1d76db` | Being worked on | Ralph | → `in-review` (ralph) |
| `in-review` | `#d876e3` | PR awaiting human review | Ralph | → done (human) or → `in-progress` (changes requested) |
| `blocked` | `#b60205` | Stuck on dependency/error | Ralph (3 failures) | → `in-progress` (after unblocking) |
| `bug` | `#d73a4a` | Something isn't working | Human | — |
| `enhancement` | `#a2eeef` | Feature request | Human | — |
| `infra` | `#0E8A16` | Infrastructure & cluster | Human | — |
| `model` | `#5319E7` | Model/architecture work | Human | — |

## 6. Environment Notes

### Local (macOS)

- Package manager: `uv`
- Run tests: `uv run pytest tests/ -k "test_name"`
- `/ralph-loop` runs directly — no SLURM, CPU-only tests

### Cluster (BWUniCluster)

- Package manager: `uv`
- CPU tests: `uv run pytest tests/<file> -x -q -k "<test>"`
- GPU tests: `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00 uv run pytest ...`
- `/slurm-submit` triggers `ralph.yml` workflow → `claude-code-action@v1` on self-hosted runner

### GPU test routing

TDD skill auto-detects GPU need by scanning test/source for `jax`, `habitat_sim`, `torch.cuda`, or `@pytest.mark.gpu`. If matched → wraps in `srun`. Otherwise → runs directly.

### Partitions

| Partition | Use | Max Time |
|-----------|-----|----------|
| `dev_gpu_h100` | Testing, TDD GPU tests | 30 min |
| `gpu_h100` | Standard GPU jobs | 48h |
| `gpu_h100_il` | Production training | 24h |
