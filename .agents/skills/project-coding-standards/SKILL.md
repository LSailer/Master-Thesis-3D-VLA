---
name: project-coding-standards
description: Apply this repo's detailed code quality rules when implementing, refactoring, or reviewing production or kept scratchpad code.
---

# Project Coding Standards

Use these standards for code that should be kept, copied into production, or used as evidence for an experiment. Throwaway prototype shells may be rough, but reusable logic should still be clear.

## Clarity and locality

- Use meaningful names for functions, variables, classes, and modules.
- Match nearby project style, naming, typing, and file structure before introducing a new pattern.
- Keep modules cohesive and loosely coupled. Prefer changes that localize behavior instead of spreading knowledge across callers.
- Extract named predicates or helper functions for hard-to-read conditionals. If the state model is still confusing, question the model rather than only rearranging branches.

## Simplicity

- Solve the current problem directly. Avoid clever patterns, speculative extension points, premature generalization, and YAGNI features.
- Avoid trivial wrappers whose body is only an optional docstring and a single `return`, unless the wrapper creates a meaningful domain abstraction or isolates a volatile dependency.
- Use SOLID as guidance where useful, but prioritize understandable, modifiable code over pattern compliance.
- Remove dead code and resolve TODOs when in scope. If a TODO must remain, make it explicit, actionable, and justified.

## Types and contracts

- Avoid untyped code and `any` unless no practical typed alternative exists.
- Prefer project/domain types and existing helpers over reimplementing local equivalents.
- Comments and docstrings should explain contracts, assumptions, inputs, outputs, shapes/dtypes, side effects, or why a non-obvious decision exists.
- If a comment only explains what straightforward code does, make the code clearer or extract a well-named function.

## Validation discipline

- Do not run lint or tests after every edit. During iteration, prefer inspection, search, syntax checks, or a narrow repro.
- Before handoff or commit of non-doc Python changes, run the narrowest relevant checks, including `python -m pylint <changed paths>` where practical.
- Fix reported errors, warnings, and infos instead of silencing them. Ask before adding scoped disables.
- CPU tests are appropriate for CPU-safe behavior. End-to-end, Habitat, VGGT, training, profiling, and GPU-marked checks must use `srun`/sbatch.

## Generated scratchpad code

- Import existing project functions, classes, constants, and types instead of reimplementing them.
- Add docstrings for reusable Python functions/classes, including shapes/dtypes and side effects when relevant.
- Record non-obvious assumptions or decisions as short `# Assumption:` or `# Decision:` comments only where they help future transfer into production.
