# PRD to Issues

Decomposes a PRD issue into vertical-slice task issues with proper dependencies, labels, and board placement.

## Trigger

User says `/prd-to-issues <issue_number>`.

## Inputs

- `prd_issue`: GitHub issue number of the PRD

## Process

### 1. Read the PRD

```bash
gh issue view <prd_issue> --json title,body,milestone,labels -q '{title: .title, body: .body, milestone: .milestone.title, labels: [.labels[].name]}'
```

Parse out:
- Acceptance criteria (the checklist)
- Module design (interfaces, dependencies)
- SLURM requirements
- Milestone

### 2. Decompose into tasks

Each task must be a **vertical slice** — independently deliverable, testable, mergeable.

Decomposition strategy:
1. One task per module/interface from the Module Design section
2. Split large modules into sub-tasks if they have >5 acceptance criteria
3. Add integration task(s) if modules interact
4. Add documentation task only if explicitly needed

For each task, determine:
- **Title**: imperative verb phrase (e.g., "Implement UNITE feature encoder")
- **Acceptance criteria**: subset of PRD criteria + any derived criteria
- **Dependencies**: which other tasks must complete first
- **Label**: `AFK` (fully automatable) or `HITL` (needs human input)
- **Compute**: CPU-only or GPU-required (from SLURM requirements)

### 3. Order by dependency

Sort tasks so that:
- Tasks with no dependencies come first
- Dependent tasks reference earlier issue numbers
- Integration tasks come last

### 4. Create issues

For each task, in dependency order:

```bash
gh issue create \
  --title "<imperative verb phrase>" \
  --label "<AFK|HITL>" \
  --label "backlog" \
  --milestone "<same as PRD>" \
  --body "$(cat <<'EOF'
## Parent PRD

Closes criteria in #<prd_issue>

## Description

<what this task implements>

## Acceptance Criteria

- [ ] <criterion 1>
- [ ] <criterion 2>
- [ ] ...

## Blocked By

- #<dep_issue_1> (if any)

## Technical Notes

- Compute: <CPU/GPU>
- Partition: <if GPU>
- Key files: `src/<path>`, `tests/<path>`
- Interface: `def func(arg: Type) -> ReturnType`
EOF
)"
```

Capture the created issue number from output for subsequent dependency refs.

### 5. Add to project board

```bash
gh project item-add 2 --owner LSailer --url <issue_url>
```

If project board is not accessible, skip this step and note it in the summary.

### 6. Output summary

Print a table:

```
Issues created from PRD #<N>: "<title>"
═══════════════════════════════════════════════════
 #   │ Title                          │ Type │ Blocked By │ Labels
─────┼────────────────────────────────┼──────┼────────────┼──────────
 <N> │ Implement X                    │ AFK  │ —          │ backlog
 <M> │ Implement Y                    │ AFK  │ #<N>       │ backlog
 <K> │ Integrate X + Y                │ HITL │ #<N>, #<M> │ backlog
═══════════════════════════════════════════════════
Total: <count> issues | AFK: <n> | HITL: <m>
```

## Rules

- Every task must have at least one testable acceptance criterion
- Tasks should be completable in 1-3 TDD sessions (~2-8 tests each)
- `AFK` label: task can be fully implemented by ralph without human input
- `HITL` label: task needs human decisions (architecture, evaluation judgments, etc.)
- All tasks get `backlog` label initially
- Milestone must match the parent PRD
- Never create circular dependencies
- If a task is GPU-dependent, note the partition in Technical Notes
- Reference the parent PRD issue in every task body
