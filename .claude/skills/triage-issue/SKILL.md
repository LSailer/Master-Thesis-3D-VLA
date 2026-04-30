---
name: triage-issue
description: Investigate a bug by exploring the codebase, find root cause, create a GitHub issue with a TDD fix plan. Use when a bug is reported, SLURM job fails, shape mismatch occurs, or any problem needs systematic investigation.
---

# Triage Issue

Investigate a reported problem, find its root cause, and create a GitHub issue with a TDD fix plan. Mostly hands-off — minimize questions to the user.

## Process

### 1. Capture the problem

Get a brief description from the user. If they haven't provided one, ask ONE question: "What's the problem you're seeing?"

Do NOT ask follow-up questions. Start investigating immediately.

### 2. Explore and diagnose

Use the Agent tool with subagent_type=Explore to deeply investigate. Find:

- **Where** the bug manifests (entry points, error messages, SLURM logs)
- **What** code path is involved (trace the flow)
- **Why** it fails (root cause, not just symptom)
- **What** related code exists (similar patterns, tests, adjacent modules)

Look at:
- Related source files and their dependencies
- Existing tests (what's tested, what's missing)
- Recent changes to affected files (`git log` on relevant files)
- Error handling in the code path
- Similar patterns elsewhere that work correctly
- `docs/wiki/` for known issues or related experiments

### 3. Identify the fix approach

Determine:
- The minimal change needed to fix the root cause
- Which modules/interfaces are affected
- What behaviors need to be verified via tests
- Whether this is a regression, missing feature, or design flaw

### 4. Design TDD fix plan

Create a concrete, ordered list of RED-GREEN cycles. Each cycle is one vertical slice:

- **RED**: A specific test that captures the broken/missing behavior
- **GREEN**: The minimal code change to make that test pass

Rules:
- Tests verify behavior through public interfaces, not implementation details
- One test at a time, vertical slices (NOT all tests first)
- Each test should survive internal refactors
- Include a final refactor step if needed
- Describe behaviors and contracts, not file paths or line numbers

**TDD applies to infrastructure code** (env wrappers, networks, replay buffer, preprocessing). For experiment code (training loops, configs), describe the fix directly without TDD framing.

### 5. Create the GitHub issue

Create using `gh issue create`. Do NOT ask the user to review first — just create it and share the URL.

```bash
gh issue create \
  --label bug \
  --title "<concise title>" \
  --body "$(cat <<'EOF'
## Problem

- **Actual**: what happens
- **Expected**: what should happen
- **Reproduce**: how to trigger it (if applicable)

## Root Cause Analysis

What was found during investigation:
- The code path involved
- Why the current code fails
- Contributing factors

## TDD Fix Plan

1. **RED**: Write a test that [expected behavior]
   **GREEN**: [Minimal change to pass]

2. **RED**: Write a test that [next behavior]
   **GREEN**: [Minimal change to pass]

**REFACTOR**: [Cleanup if needed]

## Acceptance Criteria

- [ ] Root cause is fixed
- [ ] New tests cover the failure case
- [ ] Existing tests still pass
EOF
)"
```

After creating, print the issue URL and a one-line root cause summary.
