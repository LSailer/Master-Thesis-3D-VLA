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


## Lessons learned (2026-04-11)

Issues encountered during session:
- `errors, security issues, then simplifies for clarity. Use before /qa or standalone after any code change.\n- engineer: Implement features from a plan.`
- `Error Files:**\n- `/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/output/r2dreamer-habitat-baseline/slurm-3903418.out` (4.5 KB)\n  - Cont`
- `exit code 139`
- `FAILED (exit code 139)\n127\tNodes: 1\n128\tCores per node: 8\n129\tCPU Utilized: 00:22:58\n130\tCPU Efficiency: 9.68% of 03:57:12 core-walltime\n131\`
- `error": true, "tool_use_id": "toolu_01GHgZqoozASxDYgKPhRVfDR"}]}, "uuid": "5b0f15f6-8270-4cf3-95d3-3539f729d27c", "timestamp": "2026-04-11T06:19:45.31`


## Lessons learned (2026-04-11)

Issues encountered during session:
- `error prevention", "source": "all", "limit": 5}, "caller": {"type": "direct"}}], "stop_reason": "tool_use", "stop_sequence": null, "stop_details": nul`
