---
name: research-pairing
description: Interactive research-oriented pair-programming workflow. Use when the user wants to understand a problem, explore ideas through small runnable experiments, implement vertical slices over time, and preserve durable domain understanding without a waterfall PRD.
---

# Research Pairing

Use this skill for exploratory implementation where the goal is to learn through code while keeping project understanding current.

This is not a PRD-first workflow. Treat the active issue as a living research notebook and implementation ledger, not as a fixed plan.

## Mission

Help the user move through this loop:

```text
Ask -> Restate -> Micro-experiment -> Implement -> Check -> Learn -> Decide -> Document durable knowledge
```

Optimize for:

- problem understanding before edits
- small runnable experiments
- vertical slices added over time
- frequent decision points
- durable domain understanding in `CONTEXT.md` or ADRs only when knowledge stabilizes
- pi `/tree` branches when comparing alternatives

## Start Rule

Do not start implementation immediately.

First ask 2-5 focused discovery questions unless the user already answered them in the current session. Prefer questions that clarify:

- the problem or uncertainty
- the desired outcome
- the success signal
- relevant constraints
- known files, modules, or prior attempts

If the user wants speed, ask the smallest useful question set and offer explicit assumptions.

## Context Loading

Before editing code or project docs:

1. Read the nearest `AGENTS.md`.
2. Read the active Linear issue if an issue key is present.
3. Read `CONTEXT.md` when the task touches durable project behavior, architecture, data flow, terminology, experiments, or cross-module design.
4. Read relevant ADRs when changing architectural or domain decisions.
5. Read relevant source files in full before broad edits.

## Living Issue Anchor

When an issue exists, use it as the working memory for the exploration.

The issue should contain or accumulate:

- problem / goal
- current understanding
- success signal
- open questions
- experiment log
- vertical slices discovered over time
- decisions
- durable-domain-update candidates

Do not expand it into a complete upfront plan. Add future slices only when the current slice reveals the next useful step.

After each completed micro-slice, produce a concise issue update candidate:

```md
### Experiment: <name>
Hypothesis:
Change:
Result:
Learning:
Next decision:
```

If issue-tool access is available, ask before making large issue-description rewrites. Short comments or checklist updates may be proposed directly.

## Micro-Slice Workflow

For each slice:

1. Restate the immediate goal in 1-3 sentences.
2. Name the smallest useful experiment or implementation slice.
3. Explain the intended change briefly.
4. Edit only what is needed for that slice.
5. Run the narrowest relevant check.
6. Summarize what changed, what was learned, and what decision is next.
7. Add or propose a concise issue update.

Prefer a reversible small change over a large speculative design.

## Interview While Implementing

Ask for user input only at real decision points, such as:

- naming a domain concept
- choosing behavior for an edge case
- selecting between implementation seams
- deciding whether existing behavior is intentional
- choosing whether a finding is durable enough for `CONTEXT.md` or an ADR
- choosing between two plausible experiment branches

Avoid asking for approval on every mechanical edit.

## Domain Understanding

Separate working notes from durable knowledge.

Use the issue for provisional learning.

Propose `CONTEXT.md` updates when a stable project fact emerges, such as:

- a domain term and definition
- an invariant
- a data-flow contract
- an experiment convention
- a cross-module assumption

Propose an ADR when a durable architectural decision emerges, including:

- chosen interface boundaries
- rejected alternatives
- rationale
- consequences

Ask before editing `CONTEXT.md` or ADRs.

## pi Tree Branching

Use pi `/tree` when there are competing approaches.

At branch points, tell the user:

```text
This is a good /tree branch point.
```

Keep each branch small enough to compare. When leaving a branch, summarize:

- domain assumption tested
- files touched
- result
- reason to keep or abandon the branch

## Response Format

Use this compact structure during active work:

```text
Current slice:
Change made:
Check run:
Learning:
Decision needed:
Suggested next slice:
```

When no implementation has started yet, use:

```text
Understanding so far:
Questions:
Assumptions if we proceed:
First possible micro-slice:
```
