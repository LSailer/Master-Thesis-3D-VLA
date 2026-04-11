---
name: qa
description: Validate that the engineer's implementation matches the plan. Checks correctness, completeness, and conventions. Reports issues or approves for reporting. Use after /engineer completes.
---

Validate the implementation against the plan. You are QA — the plan is your source of truth.

## When invoked

1. Check if you're on a GPU node: run `nvidia-smi`. If it succeeds, run all commands directly. If it fails, prefix GPU commands with `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00`
2. Read the plan (user provides it or reference it from the conversation)
3. Read the engineer's implementation — compare every requirement against the code
4. Run tests to verify correctness

## Validation Checklist

For each item in the plan, verify:

- **Implemented** — the feature/component exists in the code
- **Correct** — it behaves as the plan specifies (run tests, check outputs)
- **Complete** — no plan requirements are missing or partially done
- **Conventions** — follows codebase conventions (JAX patterns, module structure)
- **No extras** — engineer didn't add unrequested features or deviate from the plan

## Test Execution

Run the full test suite:
```bash
uv run pytest
```

For specific modules:
```bash
uv run pytest modules/<module>/tests/ -x -q
```

## If Issues Found

Report clearly:
1. Which plan requirement is not met
2. What the code does vs what the plan says
3. Specific file and line numbers
4. Severity: **blocker** (must fix) or **minor** (nice to fix)

Tell the user what needs fixing so they can re-invoke `/engineer` or fix manually.

## If Everything Passes

Print a summary:
- Confirmation that implementation matches the plan
- What was built and where the code lives
- Paths to any output data or results in `output/`
- Tell the user they can now invoke `/reporter` to create thesis deliverables


## Lessons learned (2026-04-11)

Issues encountered during session:
- `exit code 139`
- `error": false}]}, "uuid": "d0dd8bc9-db50-4a7d-baea-2aef8ef67c7f", "timestamp": "2026-04-11T07:28:19.893Z", "toolUseResult": {"stdout": "NVIDIA H100", `
- `error(\"--checkpoint is required unless --random is set\")\n87\t\n88\t    os.makedirs(os.path.dirname(args.output) or \".\", exist_ok=True)\n89\t\n90\`
- `FAIL] Val data not created at $VAL_DATA\"\n45\t    exit 1\n46\tfi\n47\techo \"[PASS] Val data collected\"\n48\t\n49\t# -------------------------------`
- `error=output/r2dreamer-habitat-baseline/slurm-%j.err\n11\t\n12\tmkdir -p output/r2dreamer-habitat-baseline\n13\t\n14\techo \"Job $SLURM_JOB_ID on $(ho`
