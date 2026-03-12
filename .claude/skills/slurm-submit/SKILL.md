# SLURM Submit

Helper for triggering ralph and GPU jobs from local Mac via GitHub Actions + BWUniCluster self-hosted runner.

## Trigger

User says `/slurm-submit` with optional parameters.

## Inputs

- `workflow`: workflow file to trigger (default: `ralph.yml`)
- `max_tasks`: max tasks for ralph (default: 5)
- `time_limit`: time limit in minutes (default: 120)

## Process

### 1. Trigger workflow

```bash
gh workflow run <workflow> \
  --field max_tasks=<max_tasks> \
  --field time_limit=<time_limit>
```

### 2. Get run ID

Wait a few seconds for the run to register, then:

```bash
gh run list --workflow=<workflow> --limit=1 --json databaseId -q '.[0].databaseId'
```

### 3. Monitor

```bash
gh run watch <run_id>
```

Or for non-blocking check:

```bash
gh run view <run_id> --json status,conclusion -q '{status: .status, conclusion: .conclusion}'
```

### 4. View logs on failure

```bash
gh run view <run_id> --log-failed
```

## Partition Reference

| Partition | Use Case | Max Time | Max Nodes | GPUs |
|-----------|----------|----------|-----------|------|
| `dev_gpu_h100` | Testing, validation, TDD GPU tests | 30 min | 1 | 1-4 H100 |
| `gpu_h100` | Standard GPU jobs | 48h | 1 | 1-4 H100 |
| `gpu_h100_il` | Production training (interactive-like) | 24h | 1 | 1-4 H100 |

## Common Patterns

### Run ralph (default)
```
/slurm-submit
```

### Run with custom limits
```
/slurm-submit --max_tasks 3 --time_limit 60
```

### Check status of last run
```bash
gh run list --workflow=ralph.yml --limit=1
```

### Cancel a running workflow
```bash
gh run cancel <run_id>
```

## Rules

- This skill does NOT run SLURM commands directly — it triggers GitHub Actions workflows that run on BWUniCluster
- Individual GPU tests within TDD use `srun` directly (handled by the TDD skill)
- For long training jobs, create a dedicated workflow rather than using ralph
- Always check workflow status after triggering — don't fire and forget
