# Agent loop (`scripts/agent_loop.sh`)

A tmux+srun dispatcher that picks `ready-for-agent` issues from this repo,
implements them with Claude Opus 4.7, reviews the diff, pushes commits, and
opens a PR — all surviving SSH disconnect.

> **Status**: stub. The first verification step in `.claude/plans/my-team-want-a-purrfect-curry.md`
> is an issue that asks the loop itself to fill out this doc end-to-end. If
> you're reading this and it's still a stub, that test hasn't run yet.

## Quick start

```bash
# From the BWUniCluster login node:
./scripts/agent_loop.sh                        # default 24h budget, 20 issues max
./scripts/agent_loop.sh --prd 78               # only drain children of PRD #78
./scripts/agent_loop.sh --max-issues 1 smoke   # one-issue smoke test
./scripts/agent_loop.sh --dry-run              # pick + log, do not invoke Claude
```

After launch you're attached to the tmux session. Detach with **Ctrl+b d** —
the loop keeps running. Reattach later with `tmux attach -t agent-loop`.

## Flags

| Flag            | Default     | Meaning                                              |
| --------------- | ----------- | ---------------------------------------------------- |
| `--prd N`       | _(none)_    | Restrict picker to children of PRD #N.               |
| `--max-issues N`| `20`        | Stop after N issues processed.                       |
| `--time HH:MM:SS` | `24:00:00`| SLURM `--time` for the srun reservation.             |
| `--partition X` | `gpu_h100`  | SLURM partition.                                     |
| `--dry-run`     | off         | Pick the next issue, print it, exit. Skip Claude.    |
| _(positional)_  | `agent-loop`| tmux session name.                                   |

## What the loop does, per iteration

1. `git fetch` + reset to `origin/main`.
2. Pick the lowest-numbered open issue with `ready-for-agent` and without
   `blocked` / `in-progress` / `needs-info`.
3. Determine PRD parent from the issue body's `## Parent #N` reference.
   Branch is `agent/prd-<N>` (PRD child) or `agent/issue-<M>` (orphan).
4. Flip labels: `ready-for-agent` → `in-progress`.
5. **Implement pass**: `claude -p --model claude-opus-4-7 --dangerously-skip-permissions`
   on `scripts/agent_loop_prompts/implement.md`. Must produce ≥1 commit.
6. **Review pass**: same flags on `scripts/agent_loop_prompts/review.md`.
   Writes `APPROVED` or `BLOCKED: …` to `.agent_loop/review-verdict`.
7. Push the branch.
8. Close the issue with a branch-reference comment.
9. **PR**: orphan → open immediately. PRD child → only when this PRD has
   zero open siblings remaining.

## Failure handling

Hard-stop on first failure. The failing issue keeps `in-progress` and gets
a comment containing the last 100 lines of stderr. The tmux pane prints
`LOOP HALTED: see issue #<N>` and exits 1. No further issues are consumed.

Recover by inspecting the issue + comment, fixing the root cause (in code,
prompt, or issue body), then re-launching `agent_loop.sh`.

## Troubleshooting

### `LOOP HALTED: see issue #N` in the tmux pane

The loop hit a hard failure on issue #N and called `halt()`. The diagnostic
comment (reason + last 100 lines of stderr) is on the issue itself, not in
the tmux scrollback. The issue is left labelled `in-progress`, so the picker
will skip it on the next launch.

```bash
gh issue view <N> --comments              # read the halt reason
# fix the root cause in code / prompt / issue body, commit if needed
gh issue edit <N> --remove-label in-progress --add-label ready-for-agent
./scripts/agent_loop.sh                   # relaunch
```

### Orphaned tmux session after SLURM reservation expires

When the `srun` reservation hits its `--time` limit, the inner shell dies but
the outer tmux session can linger. The pane shows `srun: error: ...` or sits
at a closed bash prompt; `tmux ls` lists the session, but `squeue -u $USER`
shows no job.

```bash
squeue -u "$USER"                         # confirm no job
tmux ls                                   # find the stale session
tmux kill-session -t agent-loop           # or whatever name was used
```

### `git push` rejected mid-iteration

The remote branch (`agent/issue-<M>` or `agent/prd-<N>`) advanced from another
source, or `main` was force-pushed under it. `halt()` will have already fired
with `git push failed`.

```bash
git fetch origin
git rebase origin/<branch>                # reconcile, then relaunch the loop
# — or, if the remote branch is junk —
git push origin --delete <branch>         # let the loop recreate it from main
```

### `.agent_loop/review-verdict` missing or unparseable

The review pass crashed before writing the verdict file, or wrote something
other than `APPROVED` / `BLOCKED: …`. The halt comment will read
`review verdict not APPROVED (got: MISSING)` or similar. The actual stderr
from the review pass is in the per-issue log:

```bash
less .agent_loop/review-<N>.log           # real error from the review claude run
cat .agent_loop/review-verdict            # whatever (if anything) was written
```

Fix the prompt or the underlying error, flip the issue back to
`ready-for-agent` (see first subsection), and relaunch.

## Files

| Path                                                | Role                                  |
| --------------------------------------------------- | ------------------------------------- |
| `scripts/agent_loop.sh`                             | Outer wrapper (tmux + srun).          |
| `scripts/agent_loop_inner.sh`                       | The loop itself (runs inside srun).   |
| `scripts/agent_loop_prompts/implement.md`           | Implement-pass prompt template.       |
| `scripts/agent_loop_prompts/review.md`              | Review-pass prompt template.          |
| `.agent_loop/`                                      | Per-issue logs + the `review-verdict`. |

## Feeding the loop

The loop only sees issues with the `ready-for-agent` label. Two paths
deliver issues into that state:

1. **`/triage`** marks an existing issue `ready-for-agent` after triage.
2. **`/to-issues` for AFK slices** — when an approved vertical slice is
   tagged `AFK` (autonomous), apply `ready-for-agent` directly instead of
   the default `needs-triage`. AFK slices are by definition fully specified
   by the parent PRD and don't need a second pass through triage. **HITL**
   slices still go through `needs-triage` — they're for a human, not the loop.

See ADR 0001 for why `ready-for-agent` is the only picker label.

## See also

- `CONTEXT.md` — glossary (`agent loop`, `PRD branch`, `halt`, `drain`).
- `docs/adr/0001-canonical-ready-for-agent-label.md` — why `ready-for-agent`.
- `docs/agents/issue-tracker.md` — gh CLI conventions.
- Issue #78 (PRD: Autoresearch) — the single-experiment ancestor of this loop.
- Issue #114 — tracks removal of legacy `scripts/run_ralph.sh`.
