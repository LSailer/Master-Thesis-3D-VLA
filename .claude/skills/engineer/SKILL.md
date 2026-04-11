---
name: engineer
description: Implement features from a plan. Receives the plan output (e.g., from Ultra Plan or plan mode) and turns it into working code. Use after planning is complete.
---

Implement the plan provided below. You are the engineer — turn the plan into working code.

## When invoked

1. Read the plan carefully — understand requirements, architecture decisions, and acceptance criteria
2. Check if you're on a GPU node: run `nvidia-smi`. If it succeeds, run all commands directly. If it fails, prefix GPU commands with `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00`
3. Review existing code in the relevant `modules/` directories for patterns and conventions
4. Check dependencies in `pyproject.toml` — install missing ones with `uv add` if needed
5. Implement step by step, following the plan's structure

## Implementation

- Work through the plan methodically, one component at a time
- Write clean, minimal code that satisfies the plan's requirements
- **Test each component incrementally** — run a quick smoke check after each deliverable, don't batch all code then test at the end
- Commit after each logical unit of work with clear commit messages
- Follow the plan — if something seems wrong, stop and ask rather than improvising
- When setting config values on third-party libraries (OmegaConf, Hydra, Habitat), verify the key exists first with a quick Python snippet
- At API boundaries (Habitat, gym, etc.), don't assume types — use defensive patterns for values that could be arrays or lists

## Before starting

Use `/skill-discovery` to check if reusable skills exist for the components you're about to build.

## After completing

Use `/delegate-task` to upload any novel implementation pattern as a reusable skill.

## Environment

- **Package manager**: uv
- **Language**: Python, JAX/Equinox preferred over PyTorch where possible

## Codebase Conventions

- Source in `modules/`, tests in `modules/*/tests/`
- JAX: `jax.numpy`, `jax.jit`, no in-place mutations, `jax.random.split` for PRNG
- Type hints on all public interfaces
- Equinox modules: `__call__` on single examples, callers vmap

## When done

Print a summary of what was implemented and where the code lives, so the user can invoke `/qa` next.


## Lessons learned (2026-04-11)

Issues encountered during session:
- `exit code 139`
- `error": false}]}, "uuid": "4dc527e5-788d-426f-9c57-c16ea8cb90e9", "timestamp": "2026-04-11T07:03:35.606Z", "toolUseResult": {"stdout": "NVIDIA H100", `
- `error(\"--checkpoint is required unless --random is set\")\n45\t\n46\t    os.makedirs(os.path.dirname(args.output) or \".\", exist_ok=True)\n47\t\n48\`
- `error": false}]}, "uuid": "9fd4fadd-6a44-44d8-9cc0-39d7310dcd96", "timestamp": "2026-04-11T07:05:15.498Z", "toolUseResult": {"stdout": "collect_expert`
- `Error:\n12\t    raise ImportError(\"habitat-sim required \u2014 install via uv sync\")\n13\t\n14\timport wandb\n15\t\n16\t\n17\tNUM_ACTIONS = 6  # STO`


## Lessons learned (2026-04-11)

Issues encountered during session:
- `error logs, identifies the root cause, generates repair patches, and verifies the fix automatically\n- **AUTO-IMPROVE:** Monitors skill performance an`
