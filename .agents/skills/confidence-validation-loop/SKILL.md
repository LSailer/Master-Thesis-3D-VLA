---
name: confidence-validation-loop
description: Evidence-based validation workflow for code changes. Use when the user asks for confidence, smoke tests, checks, validation, whether a pipeline works, or after implementing code that should be verified.
---

# Confidence Validation Loop

Use this skill to turn vague confidence into evidence-backed validation.

The goal is to reduce repeated user prompts like:

- "How confident are you?"
- "Run a smoke test."
- "Does the whole pipeline work?"
- "Check it on GPU."
- "Are the changes safe?"

## Core Rule

Do not claim confidence from reasoning alone when a narrow check is available.

If validation was not run, say so explicitly:

```text
Validation not run. Confidence is based only on code inspection.
```

## Validation Loop

After code changes, or when the user asks for confidence:

1. Identify the changed files and affected behavior.
2. State the risk surface in 1-3 bullets.
3. Run the narrowest relevant check.
4. If the check fails, fix the cause and rerun the same check.
5. If a broader smoke test is needed, propose or run the smallest smoke test that exercises the changed path.
6. Report confidence with evidence, remaining risks, and the next check that would increase confidence.

## Repository Checks

Follow project instructions first.

For this repository:

- Before editing, run:
  - `git rev-parse --show-toplevel`
  - `git status --short --branch`
  - `git worktree list`
- After Python code changes, run the narrowest required lint check:
  - `python -m pylint <changed paths>`
- Do not call lint checks tests.
- If tests are needed, run the narrowest relevant test target and name it as a test.
- Do not hide command output.

## Choosing Checks

Prefer checks in this order:

1. Static/local check for edited files.
2. Unit test that directly covers the changed behavior.
3. Minimal script or smoke command that exercises the changed pipeline path.
4. GPU or SLURM smoke only when CPU/local checks cannot validate the risk.
5. Full training/evaluation only when the user explicitly asks or the change requires it.

Ask before expensive checks, long GPU jobs, large W&B runs, or SLURM submissions unless the user already requested them.

## Confidence Report

Use this format:

```text
Confidence: <low|medium|high> (<optional percent if user asked>)
Evidence:
- <command/check and result>
- <code path inspected>
Remaining risks:
- <risk not covered by checks>
Next confidence increase:
- <smallest useful next check>
```

Guidelines:

- **High**: directly relevant checks passed and remaining risk is narrow.
- **Medium**: code inspection plus partial checks passed, but integration/GPU/data risk remains.
- **Low**: no relevant check ran, failures remain, or behavior depends on unavailable environment.

If the user asks for a percentage, give one with reasons. Avoid fake precision.

## Smoke Test Design

A smoke test should be small, fast, and targeted.

For pipeline changes, it should verify at least:

- imports and configuration construction
- representative input shape/dtype/device
- one forward/preparation step
- expected output keys/shapes/contracts
- no obvious host/device mismatch for JAX/GPU paths

For JAX/GPU checks, remember:

- account for async execution with `block_until_ready()` when timing
- distinguish host RAM, NumPy arrays, device arrays, and transfers
- check device visibility before concluding GPU behavior
- compare CPU and GPU only with the same workload and synchronization

## Stop Conditions

Stop and ask the user when:

- the next check is expensive or needs GPU/SLURM time
- validation requires credentials, datasets, or external services
- failures suggest changing intended behavior
- multiple plausible fixes exist

## Final Response

End with a compact summary:

```text
Changed:
Check run:
Result:
Confidence:
Remaining risk:
Next:
```
