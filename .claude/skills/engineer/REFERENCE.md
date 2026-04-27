# Engineer Reference

Loaded by `/engineer` on demand. Not part of the runtime workflow checklist — these are decision rules and templates the engineer consults when relevant.

## TDD vs direct implementation

**TDD (RED → GREEN tracer bullets):** anything with shape contracts or well-defined interfaces.
- env wrappers
- network modules
- replay buffers
- preprocessing pipelines

**Direct implementation:** correctness defined by experimental results, not test assertions.
- training loops
- configs
- experiment scripts
- SLURM jobs

When in doubt: if a unit test would meaningfully assert correctness without re-running the experiment, prefer TDD. If correctness only emerges after a full training run, write the code directly with smoke checks.

## TDD workflow

Derive test cases from the plan's acceptance criteria. Then vertical slices — one test, one implementation, repeat.

```
RED:   Write test for one behavior → test fails
GREEN: Write minimal code to pass → test passes
       Repeat for next behavior
```

### Rules

- **Test behavior, not implementation.** Tests exercise public interfaces. A test should survive an internal refactor without changing.
- **One test at a time.** Don't write all tests first (horizontal slicing). Each test responds to what you learned from the previous cycle.
- **Minimal code per cycle.** Only enough to pass the current test. Don't anticipate future tests.
- **No speculative features.** If the plan doesn't ask for it, don't build it.

## Design principles

- **Deep modules**: small interface, deep implementation. Fewer methods, simpler params, complex logic hidden inside.
- **Mock only at system boundaries**: Habitat/gym APIs, external services, time, randomness. Never mock your own modules.
- **Pure functions**: compute and return values rather than mutating state. Natural fit for JAX's functional style.
- **Minimal surface area**: fewer methods and parameters → fewer test cases and less coupling.

## Experiment MD template

Used by the "When done" phase. Save to `docs/wiki/experiments/<name>.md`:

```markdown
# <Experiment Name>

**Status**: implemented
**Date**: YYYY-MM-DD
**Tags**: #relevant #tags
**Wandb**: <run URL if available>
**SLURM Job ID**: <job id if available>

## Setup
What was tested and why. Hypothesis in one sentence.

## Changes
What changed compared to the previous run.

## Configuration
Key config values, hyperparameters, environment settings.
```

The reporter (auto-pipeline Phase 4) will append a `## Results` section after the run completes. Leave that section out if the run hasn't happened yet — the reporter creates it.
