---
name: bwunicluster-gpu-srun
description: Run short GPU commands and JAX/Python experiments on bwUniCluster with srun. Use when the user asks to run a GPU check, GPU smoke command, benchmark, or experiment on the cluster.
compatibility: bwUniCluster Slurm environment with GPU partitions such as dev_gpu_h100, gpu_h100_short, gpu_a100_short, and a project .venv containing Python/JAX.
---

# bwUniCluster GPU srun

Use this skill to run short GPU commands on bwUniCluster quickly and reproducibly.

## Core rules

- Use `srun` for interactive GPU execution.
- Never request more than `--time=00:30:00`.
- Prefer the project's virtual environment: `.venv/bin/python`.
- Do not run lint checks for benchmark/experiment requests unless the user explicitly asks.
- Include `hostname`, `nvidia-smi`, and the actual Python command in the `srun` payload so the result proves which GPU and environment were used.
- Save the command and result under `scratchpad/checks/` when working in this repository.

## Default H100 command

Use this template for quick GPU Python scripts:

```bash
srun --partition=dev_gpu_h100 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=4G \
  --gres=gpu:1 \
  --time=00:30:00 \
  bash -lc 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; .venv/bin/python <script-or-command>'
```

For a Python one-liner, use:

```bash
srun --partition=dev_gpu_h100 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=4G \
  --gres=gpu:1 \
  --time=00:30:00 \
  bash -lc 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; .venv/bin/python - <<"PY"
import jax
print("jax", jax.__version__)
print("devices", jax.devices())
print("gpu_devices", jax.devices("gpu"))
PY'
```

## Fast preflight

Before using GPU time, check the local environment when useful:

```bash
.venv/bin/python - <<'PY'
import importlib.util
print('jax_available', importlib.util.find_spec('jax') is not None)
PY
```

If `.venv/bin/python` has JAX but system `python` does not, always use `.venv/bin/python` inside `srun`.

## Choosing a partition

Start with:

- `dev_gpu_h100` for short H100 checks and experiments.

If the partition fails or a different GPU is requested, inspect available partitions:

```bash
scontrol show partition
```

Known useful partitions seen on this cluster:

- `dev_gpu_h100`: one H100 node, development queue, max 30 minutes.
- `gpu_h100_short`: H100 short queue, max 30 minutes.
- `gpu_a100_short`: A100 short queue, max 30 minutes.
- `dev_gpu_a100_il`: one A100 development node, max 30 minutes.
- `gpu_h100`: longer H100 queue; only use when the user explicitly wants a longer job.
- `gpu_a100_il` / `gpu_h100_il`: IL GPU queues; use only when appropriate for the user's account/location.

If `srun` reports `No partition specified`, retry with `--partition=dev_gpu_h100`.
If it reports `Invalid partition name`, run `scontrol show partition` and select one of the listed GPU partitions.

## JAX GPU timing rules

For JAX benchmarks:

- Warm up `jax.jit` functions once before timing.
- Use `.block_until_ready()` on outputs before stopping timers.
- Keep storage bytes and runtime separate in reports.
- Report the visible device, e.g. `cuda:0`, and the GPU model from `nvidia-smi`.

Minimal timing pattern:

```python
warmup = fn(*args)
warmup.block_until_ready()

start = time.perf_counter()
result = fn(*args)
result.block_until_ready()
elapsed = time.perf_counter() - start
```

For pytrees, block all array leaves:

```python
for leaf in jax.tree_util.tree_leaves(result):
    block = getattr(leaf, "block_until_ready", None)
    if callable(block):
        block()
```

## Result recording

When running from this repository, record the command and output in a Markdown file:

```text
scratchpad/checks/<short-name>_gpu_srun_result.md
```

Include:

- exact `srun` command
- GPU model and node
- script/benchmark output
- short interpretation
- whether the command completed or failed

## Common failures

### JAX is missing

Symptom:

```text
ModuleNotFoundError: No module named 'jax'
```

Fix: use `.venv/bin/python`, not system `python`.

### No local GPU tooling

Symptom on login node:

```text
nvidia-smi: command not found
```

Fix: run inside `srun`; GPU tooling may only be visible on GPU nodes.

### CPU binding warning

Symptom:

```text
hwloc_set_cpubind() failed: Invalid argument
```

Usually this is a runtime/cluster binding warning. If the benchmark completes and JAX reports `cuda:0`, do not treat it as a failed GPU run.
