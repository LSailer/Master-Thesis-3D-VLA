# Write a PRD

Interactive 5-step process to produce a Product Requirements Document for a thesis module, then publish it as a GitHub issue.

## Trigger

User says `/write-a-prd` with a topic or problem area.

## Inputs

- `topic`: brief description of what needs building

## Process

### Step 1 — Problem Exploration (interactive)

Ask the user (all at once, not one-by-one):

- What problem does this solve? What happens if we don't build it?
- What's the desired outcome / success metric?
- Any hard constraints? (hardware, time, data, dependencies)
- Which thesis milestone does this belong to?

Wait for answers before proceeding.

### Step 2 — Codebase Verification

Read relevant existing code to understand:

- What already exists that relates to this
- What interfaces/abstractions are in place
- What test patterns are used
- What dependencies are already declared

Use `Glob`, `Grep`, `Read` to explore `src/`, `tests/`, `pyproject.toml`.

Share findings with the user: "Here's what I found in the codebase..."

### Step 3 — Design Interview (interactive)

Based on steps 1-2, ask targeted clarifying questions:

- Architecture trade-offs (e.g., "Should X be a separate module or extend Y?")
- Scope boundaries (e.g., "Do we need Z in this PRD or is that a follow-up?")
- SLURM resource needs (e.g., "Training on H100 — how many GPUs, how long?")
- Evaluation approach (e.g., "Compare against which baseline?")

Wait for answers before proceeding.

### Step 4 — Module Design

Break the solution into modules with clear interfaces:

- For each module: name, responsibility, public API (function signatures / class interfaces)
- Dependency graph between modules
- Which modules need GPU, which are CPU-only
- Data flow diagram (text-based)

Present to user for feedback. Iterate if needed.

### Step 5 — PRD Generation

Write the full PRD using the template below, then create a GitHub issue:

```bash
gh issue create \
  --label prd \
  --milestone "<milestone>" \
  --title "PRD: <title>" \
  --body "$(cat <<'EOF'
<prd_content>
EOF
)"
```

## PRD Template

```markdown
## Problem Statement

<1-3 sentences. What problem, why it matters for the thesis.>

## Research Questions

- RQ1: <question this module helps answer>
- RQ2: ...

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

### Module: `<name>`
...

## Data Flow

```
input → module_a → module_b → output
                 ↘ module_c ↗
```

## SLURM Requirements

| Job Type | Partition | GPUs | Time | Notes |
|----------|-----------|------|------|-------|
| Training | gpu_h100_il | 1-4 | 6-24h | ... |
| Eval | dev_gpu_h100 | 1 | 30min | ... |
| Tests | dev_gpu_h100 | 1 | 10min | ... |

## Evaluation Criteria

- **Primary metric**: <e.g., SPL, Success Rate>
- **Baseline**: <what we compare against>
- **Statistical test**: <if applicable>

## Milestones

- Belongs to: <month milestone name>
- Estimated effort: <days/weeks>

## Acceptance Criteria

- [ ] <testable criterion 1>
- [ ] <testable criterion 2>
- [ ] ...

## Dependencies

- **Blocked by**: <other PRDs or issues>
- **Blocks**: <downstream work>
- **External**: <datasets, model weights, cluster access>
```

## Rules

- Every acceptance criterion must be testable (automatable preferred)
- SLURM requirements must specify partition and time estimate
- Module interfaces must include type signatures
- PRD scope should be achievable in 1-2 weeks of implementation
- Always check the codebase before designing — don't reinvent what exists
