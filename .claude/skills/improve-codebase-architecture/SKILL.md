---
name: improve-codebase-architecture
description: Explore codebase for architectural friction, surface shallow modules and coupling, then propose deep-module refactors as GitHub issue RFCs. Use when codebase feels tangled, hard to test, or hard to navigate.
---

# Improve Codebase Architecture

Explore the codebase, surface architectural friction, and propose module-deepening refactors as GitHub issue RFCs.

A **deep module** (Ousterhout, "A Philosophy of Software Design") has a small interface hiding a large implementation. Deep modules are more testable, more navigable, and let you test at the boundary instead of inside.

## Process

### 1. Explore the codebase

Use the Agent tool with subagent_type=Explore to navigate organically. Do NOT follow rigid heuristics — note where you experience friction:

- Where does understanding one concept require bouncing between many small files?
- Where are modules so shallow that the interface is nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called?
- Where do tightly-coupled modules create integration risk in the seams between them?
- Which parts of the codebase are untested, or hard to test?

The friction you encounter IS the signal.

### 2. Present candidates

Present a numbered list of deepening opportunities. For each candidate, show:

- **Cluster**: which modules/concepts are involved
- **Why they're coupled**: shared types, call patterns, co-ownership of a concept
- **Dependency category**: one of the four categories below
- **Test impact**: what existing tests would be replaced by boundary tests

Do NOT propose interfaces yet. Ask: "Which of these would you like to explore?"

### 3. User picks a candidate

### 4. Frame the problem space

Before spawning sub-agents, write a user-facing explanation:

- The constraints any new interface would need to satisfy
- The dependencies it would need to rely on
- A rough illustrative code sketch to make the constraints concrete — this is not a proposal, just grounding

Show this to the user, then immediately proceed to Step 5. The user reads while sub-agents work.

### 5. Design multiple interfaces

Spawn 3+ sub-agents in parallel using the Agent tool. Each must produce a **radically different** interface for the deepened module.

Prompt each sub-agent with a separate technical brief (file paths, coupling details, dependency category, what's being hidden). Give each a different design constraint:

- Agent 1: "Minimize the interface — aim for 1-3 entry points max"
- Agent 2: "Maximize flexibility — support many use cases and extension"
- Agent 3: "Optimize for the most common caller — make the default case trivial"
- Agent 4 (if applicable): "Design around the ports & adapters pattern for cross-boundary dependencies"

Each sub-agent outputs:

1. Interface signature (types, methods, params)
2. Usage example showing how callers use it
3. What complexity it hides internally
4. Dependency strategy (how deps are handled)
5. Trade-offs

Present designs sequentially, then compare in prose.

After comparing, give your own recommendation: which design is strongest and why. If elements from different designs combine well, propose a hybrid. Be opinionated.

### 6. User picks an interface (or accepts recommendation)

### 7. Create GitHub issue

Create a refactor RFC as a GitHub issue:

```bash
gh issue create \
  --label refactor \
  --title "RFC: Deepen <module name>" \
  --body "$(cat <<'EOF'
## Problem

<What's shallow/coupled and why it matters>

## Proposed Interface

<Chosen design with signature and usage example>

## What It Hides

<Complexity moved behind the interface>

## Dependency Strategy

<How external deps are handled — see categories below>

## Testing Approach

<What boundary tests replace, what internal tests can be deleted>

## Implementation Notes

<Suggested order, risks, migration steps>
EOF
)"
```

Do NOT ask the user to review before creating — just create it and share the URL.

## Dependency categories

When analyzing how a module relates to its dependencies, classify each:

1. **In-process**: pure computation, no I/O. Always deepenable — just move it behind the interface.
2. **Local-substitutable**: has test stand-ins available (e.g., in-memory DB, test fixtures). Deepen with injected dependency.
3. **Remote but owned**: your service, but across a boundary. Use ports & adapters — inject the transport layer.
4. **True external**: third-party API you don't control. Mock at the boundary only.

## Rules

- Do NOT propose interfaces before the user picks a candidate
- Do NOT implement — this skill produces RFCs, not code
- Enforce radical difference between sub-agent designs
- The value is in the exploration and comparison, not in generating a single answer
