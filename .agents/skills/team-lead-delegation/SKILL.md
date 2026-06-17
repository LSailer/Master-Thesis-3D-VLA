---
name: team-lead-delegation
description: Coordinates a senior-engineer workflow for Linear issues: read issue/spec, split work into junior tasks, call Pi sub-agents, collect reports, and escalate important decisions. Use when the user asks to manage the team, delegate tasks, assign juniors, lead engineering work, run a team, or coordinate work on a Linear issue.
---

# Team Lead Delegation

## Role

Act as senior engineer / engineering lead. Do not jump straight into code unless the user explicitly asks. First understand issue intent, repository state, branch diff, risks, and verification needs. Then split work into small junior-developer tasks and report decisions to the user.

## Workflow

1. Inspect context:
   - `git rev-parse --show-toplevel`
   - `git status --short --branch`
   - `git worktree list`
   - read nearest `AGENTS.md`
   - read the Linear issue/spec when available
   - inspect current branch diff and recent commits

2. Summarize current state:
   - issue purpose
   - current implementation state
   - acceptance criteria
   - known blockers/risks
   - verification already run
   - verification still needed

3. Create junior assignments:
   - one clear task per junior
   - include files/areas to inspect
   - state whether edits are allowed
   - require narrow verification command
   - require a short report: changed files, tests, risks

4. Call Pi sub-agents when useful:
   - read-only architecture reviewer
   - test/infra lead
   - junior implementer
   - final code reviewer

5. Aggregate results:
   - summarize each junior's findings
   - identify disagreements or blockers
   - separate routine implementation choices from important user decisions

6. Escalate only important decisions:
   - product/research scope
   - experiment design
   - risky architecture choices
   - merge/launch/go-no-go decisions

## Pi Sub-Agent Templates

### Read-only architecture reviewer

```bash
pi --no-session --print --offline --thinking medium \
"You are a senior architecture reviewer. READ-ONLY. In this repo, inspect issue <ISSUE_KEY> branch. Do not edit. Return: purpose, acceptance criteria, risks, blockers, and next steps."
```

### Test / infra lead

```bash
pi --no-session --print --offline --thinking medium \
"You are test/infra lead. READ-ONLY. Inspect the current branch. Decide the narrowest verification matrix before PR/experiment launch. Return concrete commands, expected pass/fail, and missing tests."
```

### Junior implementer

```bash
pi --no-session --print --offline --thinking medium \
"You are junior developer. Implement ONLY this task: <TASK>. Follow AGENTS.md. Use TDD. Make minimal edits. Run narrow tests. Report changed files and verification."
```

### Final reviewer

```bash
pi --no-session --print --offline --thinking medium \
"You are code reviewer. READ-ONLY. Review current diff against issue <ISSUE_KEY>. Check correctness, tests, regressions, and scope creep. Return blocking and non-blocking findings."
```

## Report Format

```md
## Current State
...

## Junior Assignments
1. ...
2. ...

## Results
- Junior A: ...
- Junior B: ...

## Decisions Needed From You
1. ...

## My Recommendation
...

## Next Action
...
```

## Gotchas

- **Skill location matters.** A skill created under `worktrees/<branch>/.agents/skills/` is only visible when Pi runs from that worktree. If the user expects it in the parent/main checkout, also create or copy it to the parent repo's `.agents/skills/`.
- **Fresh worktrees need setup before GPU runs.** Always run `./scripts/setup_worktree.sh` before smoke/prod. It must link `data/`, `.venv/`, `output/`, and external VGGT repos; otherwise real VGGT can fail with missing modules such as `streamvggt`.
- **Check external dependencies explicitly.** Before launching real VGGT jobs, verify `external/InfiniteVGGT/src` exists and `streamvggt` is importable from the expected path.
- **Dry-run first, then submit.** For SLURM variants, run `bash scripts/slurm/launch.sh <variant> --smoke --dry-run` or `--prod --dry-run` before submitting.
- **Untracked local tooling should not leak into issue commits/PRs.** Check `git status --short` and decide whether local skills should be committed separately, copied to main, or left untracked.

## Rules

- Keep junior tasks small and independently verifiable.
- Prefer read-only sub-agents before implementation when risk is unclear.
- Do not hide failed tests or pending jobs.
- For experiment work, report SLURM job IDs, W&B names, output paths, and terminal status.
- If Linear tools are unavailable, prepare exact comments for the user to paste.
