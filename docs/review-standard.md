# Review Standard

This document defines how review automation decides whether a pull request may be fixed, merged, or escalated to a human.

## Correctness

The linked Linear issue is the primary source of correctness. A pull request is correct only when it satisfies the Linear issue acceptance criteria and does not violate repository tests, documentation, or this review standard.

If the Linear issue and repository expectations disagree, the review automation must not merge. It must create a `needs-human` follow-up in Linear.

The automation is sure that a pull request is correct only when all of these are true:

- Exactly one linked Linear issue is found.
- The Linear issue has explicit acceptance criteria.
- The pull request diff maps directly to those criteria.
- The pull request introduces no unrelated behavior.
- Required CI checks pass.
- Required local validation commands pass, if available.
- The risk tier is low or medium.
- No `review: hold` command exists in Linear.
- No high-risk file or permission boundary is touched.
- The automation can write a short evidence comment explaining why the pull request satisfies the issue.

## Risk Tiers

Low-risk pull requests may be automatically merged when they satisfy the correctness standard and required checks pass.

Medium-risk pull requests may be automatically merged when they satisfy the correctness standard and required checks pass. Experiment and training code is medium risk when the Linear issue is clear and validation passes.

High-risk pull requests must not be automatically merged.

High-risk pull requests include changes to automation authority, repository infrastructure, authentication, secrets, compatibility formats, ambiguous Linear criteria, or weakened validation.

## Linear Commands

Review automation reads commands from Linear, not pull request comments.

`review: fix` authorizes the automation to attempt a medium-risk fix.

`review: merge` explicitly authorizes merge when the pull request satisfies this review standard and is low or medium risk, but it is not required for normal automatic merge.

`review: hold` blocks automatic merge.

## Linear Linking

The automation identifies the linked Linear issue from the pull request title or branch name.

Accepted formats include:

- `3D-123: short title`
- `issue/3D-123-short-title`
- `3d-123-short-title`

Exactly one Linear issue key must be found. If no key is found, the automation must not merge. If multiple keys are found, the automation must not merge unless the pull request body explicitly marks one as primary.

## Model

Review automation with merge authority uses `gpt-5.5`.

Review automation starts with `model_reasoning_effort = "medium"`. Increase effort only if review results show missed defects, weak evidence comments, or unnecessary escalations.

Fast helper agents may use `gpt-5.4-mini` for mechanical checks, but they do not have merge authority.

## Runtime

Review automation runs first as a local or HPC tmux loop. GitHub Actions cron may be added later as a trigger, but it is not the initial runtime.

The loop reviews pull requests in isolated Git worktrees, not in the main checkout. Worktrees live under `.review/worktrees/pr-<number>` and are removed after merge or escalation.

The loop reviews open pull requests with exactly one Linear key in the pull request title or branch name. It does not require an `auto-review` label.

The loop skips draft pull requests, pull requests with unresolved human requested-changes reviews, and pull requests whose linked Linear issue contains `review: hold`.

The loop should merge at most one pull request per iteration.

The loop polls every 15 minutes.

## Merge Method

Review automation uses squash merge for low- and medium-risk pull requests.

The squash merge title must include the Linear key. The automation may delete the source branch after merge if GitHub allows it.

The automation must not force push. If the branch cannot be updated cleanly against main, the automation must create a `needs-human` follow-up instead of merging.

## Follow-ups

Correctness defects that block the pull request become Linear sub-issues under the linked issue and are labeled `needs-human`.

Medium-risk fixes that are not explicitly authorized become Linear sub-issues under the linked issue and are labeled `needs-human`.

Feature ideas found during review become separate Linear issues labeled `needs-human`.

Feature ideas are created only when they are actionable and outside the linked issue's acceptance criteria. The issue must include evidence from the pull request, why the idea is out of scope, and a suggested acceptance criterion.

## Completion

After the automation merges a pull request, it must comment on the linked Linear issue with the merge result and move the Linear issue to Done.

## Validation

Docs-only changes do not require tests, but links and Markdown should be syntactically sane.

Python unit changes must run through `srun` on a GPU partition and then execute the relevant `uv run pytest ... -x -q` command:

```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 uv run pytest <test-selection> -x -q
```

GPU-marked tests use:

```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 uv run pytest <test-selection> -m gpu -v
```

R2Dreamer, VGGT, training, or experiment changes must run focused unit tests plus a smoke command when one exists for the changed path. Slurm launcher changes must include:

```bash
uv run pytest tests/slurm/test_launch.py -v
```

Training variant changes should use the relevant smoke launcher:

```bash
bash scripts/slurm/launch.sh <variant> --smoke --dry-run
bash scripts/slurm/launch.sh <variant> --smoke
```

GitHub Actions or automation changes are high risk and must not be automatically merged. The automation should still run syntax or static checks when available.

Unknown change types must not be automatically merged unless CI provides enough evidence and the completion comment explains why that evidence is sufficient.

If required CI is pending, the automation waits until a later loop iteration. If CI failed, the automation inspects the failure logs and decides whether the defect is low-risk fixable. If CI is missing for a required validation category, the automation runs local validation. If CI passed but required GPU validation is not represented in CI, the automation runs local GPU validation before merge.

## Open Questions

- Which exact validation commands are required for docs, tests, training code, and automation changes?
