---
name: plan
description: Full planning pipeline — interview relentlessly about a design, write a PRD as GitHub issue, then break into vertical-slice phases saved locally. Always runs all three phases. Use when starting a new feature, experiment, or architectural change.
---

# Plan

Three-phase planning pipeline. Always runs all phases in order — no skipping.

## Phase 1 — Grill

Interview the user relentlessly about every aspect of the plan until reaching shared understanding. Walk down each branch of the design tree, resolving dependencies one-by-one.

Rules:
- Ask **one question at a time**. Wait for the answer before continuing.
- For each question, provide your recommended answer.
- If a question can be answered by exploring the codebase, explore instead of asking.
- Read `docs/wiki/` for existing knowledge before asking questions that may already be answered.

### Web-backed research

Proactively use WebSearch to strengthen your questions and recommendations:

- **Uncertainty**: When unsure about best practices or trade-offs, search the web before asking. Do not guess.
- **Common patterns**: Search for established patterns, conventions, and prior art. Cite what you find.
- **Pitfalls**: Search for known issues or failure modes others have encountered.
- **Research context**: For ML topics, search for relevant papers, benchmarks, or comparisons.

Integrate findings into your questions naturally — don't dump raw search results.

### Gate

All branches of the design tree are resolved. User confirms shared understanding.

## Phase 2 — Document

Write a PRD and publish it as a GitHub issue.

### Process

1. **Explore the codebase** — read relevant `modules/`, tests, `pyproject.toml` to understand what exists
2. **Design modules** — break the solution into modules with clear interfaces, dependency graph, data flow
3. **Write the PRD** using the template below
4. **Create GitHub issue**:
   ```bash
   gh issue create \
     --label prd \
     --title "PRD: <title>" \
     --body "$(cat <<'EOF'
   <prd_content>
   EOF
   )"
   ```

### PRD Template

```markdown
## Problem Statement

<1-3 sentences. What problem, why it matters for the thesis.>

## Research Questions

- RQ1: <question this module helps answer>

## Proposed Solution

<High-level approach. 1-2 paragraphs.>

## Module Design

### Module: `<name>`
- **Responsibility**: <what it does>
- **Interface**:
  ```python
  def function_name(arg: Type) -> ReturnType:
      """Docstring."""
  ```
- **Dependencies**: <other modules, external libs>
- **Compute**: CPU / GPU (partition: <name>, est. time: <X>)

## Data Flow

```
input → module_a → module_b → output
```

## SLURM Requirements

| Job Type | Partition | GPUs | Time | Notes |
|----------|-----------|------|------|-------|
| Training | gpu_h100_il | 1-4 | 6-24h | ... |
| Eval | dev_gpu_h100 | 1 | 30min | ... |

## Evaluation Criteria

- **Primary metric**: <e.g., SPL, Success Rate>
- **Baseline**: <what we compare against>

## Acceptance Criteria

- [ ] <testable criterion 1>
- [ ] <testable criterion 2>
```

### Gate

PRD is published as a GitHub issue. User confirms requirements are locked.

## Phase 3 — Slice

Break the PRD into vertical-slice phases saved as a local plan file.

### Process

1. **Identify durable architectural decisions** — key data models, third-party boundaries, observation spaces, action spaces. These go in the plan header.
2. **Draft vertical slices** — each phase is a thin slice cutting through ALL layers end-to-end (env wrapper → encoder → world model → training → evaluation → logging). NOT horizontal slices of one layer.
3. **Quiz the user** on granularity — too coarse or too fine? Iterate until approved.
4. **Write the plan file** to `docs/plans/<feature>.md`

### Vertical slice rules

- Each slice delivers a narrow but COMPLETE path through every layer
- A completed slice is runnable and verifiable on its own
- Prefer many thin slices over few thick ones
- Include acceptance criteria for each phase

### Plan template

```markdown
# Plan: <Feature Name>

> Source PRD: #<issue-number>

## Architectural decisions

- **Observation space**: ...
- **Action space**: ...
- **Key models**: ...

---

## Phase 1: <Title>

### What to build

A concise description of this vertical slice end-to-end.

### Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2

---

## Phase 2: <Title>

### What to build

...

### Acceptance criteria

- [ ] ...
```

### Gate

Plan file written. User confirms phases. Ready for `/engineer`.

## Rules

- Every acceptance criterion must be testable
- SLURM requirements must specify partition and time estimate
- Module interfaces must include type signatures
- PRD scope should be achievable in 1-2 weeks
- Always check the codebase before designing — don't reinvent what exists
