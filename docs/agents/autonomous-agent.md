# Autonomous Agent Workflow

The autonomous agent polls Linear team `3D-WM-ObjectNAV` for implementation issues and publishes GitHub pull requests as linked implementation artifacts.

## Queue Selection

- Pick Linear issues with label `ready-for-agent`.
- Eligible issues must be in Linear state `Todo` or `Backlog`.
- Do not pick blocked issues.
- Prefer higher-priority issues; tie-break by oldest creation date.
- When an issue is picked, move it to `In Progress` and remove `ready-for-agent`.

## Run Limits

- The cron accepts `max_tasks`.
- Default `max_tasks` is `1`.
- A run may process up to `max_tasks` eligible issues, stopping earlier when the queue is empty or the run budget is exhausted.
- The first implementation claims exactly one issue per poll, even though `max_tasks` remains an explicit option.

## Coding Agent

- Use GPT-5.3 Codex Spark.
- Use high effort by default.
- Use the TDD skill for implementation work.
- Configure the model and effort in the scheduler/launcher code, not in the per-issue prompt.
- Launch Codex with `/goal` for the claimed Linear issue.
- The goal objective should be to complete the Linear issue according to its acceptance criteria, create the PR, and update Linear, or escalate with the documented blocker workflow.
- Pass Codex the Linear issue reference and workflow rules; Codex should fetch the issue details itself instead of relying on duplicated issue text from the scheduler.
- The launch prompt should name the Linear issue key only, require Codex to fetch the issue itself, require the TDD skill, and require either PR creation or the documented `needs-human` escalation.
- Use short `srun` jobs on `dev_gpu_h100` for GPU validation, with a maximum runtime of 30 minutes.
- Use the production GPU partition for long-running training or experiment runs.
- For now, if required GPU validation is still pending or cannot complete inside the short validation budget, escalate with `needs-human` instead of silently treating the task as done.
- If useful code changes exist and non-GPU tests passed, open a draft PR before escalating required pending GPU validation.
- The draft PR and Linear comment must include the pending SLURM job ID when available.

## Scheduling

- Poll every 20 minutes.
- On each poll, check Linear for eligible issues.
- Start a coding agent only when an eligible issue exists.
- Do not use active agent runs as the primary scheduler gate; use Linear state and blocker relationships so independent issues can be worked separately.
- The first implementation runs from a local cron/tmux setup.
- Keep the scheduler boundary narrow enough that GitHub Actions with a self-hosted runner can replace local cron/tmux later.
- For each claimed issue, start a detached tmux session when possible.
- Name sessions deterministically from the Linear issue key, for example `agent-3d-123`.
- Write local scheduler and Codex logs under `.agent/logs/<linear-key>/`.
- If tmux cannot start after an issue is claimed, move the issue back to `Todo`, remove `ready-for-agent`, add `needs-human`, and comment with the scheduler failure.
- If Codex exits non-zero without creating a PR or documented blocker, move the original issue to `Todo`, keep `ready-for-agent` removed, add `needs-human`, and comment with the exit code and tmux log path.

## Credentials

- The local scheduler requires `LINEAR_API_KEY` in its environment.
- Do not store Linear API keys in the repository.
- If `LINEAR_API_KEY` is missing, the scheduler must fail before claiming any issue.

## Human Blockers

- If the agent cannot continue, create or link a blocking subissue with label `needs-human`.
- When Linear supports it, make the `needs-human` issue both a child/subissue of the blocked issue and a blocker of the blocked issue.
- Leave the parent issue in its current state unless another rule says to move it; the blocker relation is the source of truth that keeps it out of the agent queue.
- When all blocking `needs-human` subissues are completed, requeue the parent automatically by moving it to `Todo` and keeping or re-adding `ready-for-agent`.
- If the failure is a generic agent failure with no clear actionable human blocker, do not create a child blocker issue. Move the original issue to `Todo`, remove `ready-for-agent`, add `needs-human`, and comment with the failure reason and logs.

## Follow-Up Issues

- The agent may create new `ready-for-agent` subissues for clearly separable follow-up work.
- Follow-up subissues must not remove required acceptance criteria from the current issue unless that work is blocked.

## Successful Completion

- When the agent creates a PR, move the Linear issue to `In Review`.
- Remove the `ready-for-agent` label.
- Add a Linear comment with the PR URL and verification summary.
- The PR title or body must reference the Linear issue key/name.
- Use branch names in the form `<linear-key>-<short-slug>`, for example `3d-123-add-linear-cron-runner`.
- Use a fresh git worktree per claimed issue, named from the Linear key and slug, for example `worktrees/3d-123-add-linear-cron-runner`.
- Keep worktrees after PR creation because long-running SLURM or training jobs may still depend on that checkout.
