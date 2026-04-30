---
name: engineer
description: Implement features from a scoped spec, using TDD for infrastructure code (env wrappers, networks, replay buffers) and direct implementation for training loops and configs. Use when a /grill-me recap, GitHub issue, or direct brief is ready and the user wants to turn it into committed code on the current branch.
---

# Engineer

## Quick start

> **User:** `/engineer #93`
> **You:** "Read issue #93 — DreamerV3 `act_entropy=3e-4` fix. Single phase: edit `modules/r2dreamer/agent.py:147` + add regression test in `tests/test_actor.py`. This is a config tweak with no shape contract — direct fix + targeted test, not full TDD. Confirm path and I'll implement?"

Receive a GitHub issue, /grill-me recap, or direct brief and turn it into committed code. TDD methodology, design principles, and the experiment-MD template are in [REFERENCE.md](REFERENCE.md).

## Workflow

### Setup

- [ ] Read the spec — requirements, architecture decisions, acceptance criteria
- [ ] Run `nvidia-smi`. If unavailable, prefix GPU commands with `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00`
- [ ] Read `docs/wiki/` for context on related experiments and methods
- [ ] Skim relevant `modules/` for existing patterns and conventions
- [ ] Check `pyproject.toml`; install missing deps with `uv add`

### Per phase — Implement → Refactor → Commit → Review

For each phase in the plan:

- [ ] **Implement.** Decide TDD vs direct (see REFERENCE.md). TDD: one test → one implementation → repeat. Direct: minimum that satisfies the spec + smoke check.
- [ ] **Verify config keys exist** before setting them on third-party libraries (OmegaConf, Hydra, Habitat). At API boundaries (Habitat, gym), use defensive types — values may be array or list.
- [ ] **Refactor only at GREEN.** Look for duplication, long methods, shallow modules, feature envy. Run tests after each step. Never refactor while RED.
- [ ] **Commit.** `feat(<module>): <what this phase delivers>` — one commit per phase, stage only this phase's files.
- [ ] **`/review`.** Fix blockers, amend the phase commit. Goal: one clean commit per phase.
- [ ] **Next phase only after review passes.**

## Codebase conventions

- Source in `modules/`, tests in `modules/*/tests/`
- Package manager: `uv`. Run Python via `uv run python …`
- JAX preferred over PyTorch where possible; use `jax.numpy`, `jax.jit`, `jax.random.split`. No in-place mutation.
- Equinox modules: `__call__` on single examples, callers `vmap`
- Type hints on all public interfaces

## When done

- [ ] Create `docs/wiki/experiments/<name>.md` using the template in [REFERENCE.md](REFERENCE.md)
- [ ] Add entry under Experiments in `docs/wiki/index.md`
- [ ] Append a one-liner to `docs/wiki/log.md`:
  ```
  ## [YYYY-MM-DD] ingest | <Experiment Name> | source: /engineer
  <Brief description>. Created experiments/<name>.md. Updated index.
  ```
- [ ] Print a summary of what was implemented and where the code lives
