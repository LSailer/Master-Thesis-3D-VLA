# Master Thesis: VLA + 3D Semantic Scene Understanding

Augmenting Vision-Language-Action models with UNITE 3D features for object navigation in HM3D (Habitat).

## Slides

https://LSailer.github.io/Master-Thesis-3D-VLA/



## GPU Execution

This project runs on BWUniCluster (SLURM). GPU access requires `srun` — never run GPU code directly.

**GPU tests:**
```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00 uv run pytest tests/<file> -x -q -k "<test>"
```

**General GPU commands:**
```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 <command>
```

**When to use `srun`:** if the code imports `jax`, `habitat_sim`, `torch.cuda`, or uses `@pytest.mark.gpu`, it needs GPU → wrap with `srun`.

**Partitions:**
| Partition | Use Case | Max Time |
|-----------|----------|----------|
| `dev_gpu_h100` | Testing, validation, quick experiments | 30 min |
| `gpu_h100` | Standard GPU jobs, training | 48h |


## Slurm Launcher

Use the YAML-backed launcher in [`scripts/slurm/`](scripts/slurm/) for current
training, smoke, and production jobs. Each variant lives in
`scripts/slurm/configs/<variant>.yaml`, and
[`scripts/slurm/launch.sh`](scripts/slurm/launch.sh) renders and submits the
matching sbatch script.

```bash
# Render only; submits nothing.
bash scripts/slurm/launch.sh l1_vggt --dry-run
bash scripts/slurm/launch.sh l1_vggt --smoke --dry-run

# Submit jobs.
bash scripts/slurm/launch.sh l1_vggt --smoke
bash scripts/slurm/launch.sh l1_vggt --prod
bash scripts/slurm/launch.sh l1_vggt --smoke-then-prod

# Sweep variants and override config env values.
bash scripts/slurm/launch.sh l{1,2,3,4}_vggt --smoke
bash scripts/slurm/launch.sh l1_vggt --env WANDB_MODE=offline --smoke
```

Add new experiment launchers by copying an existing YAML config, editing the
small variant-specific delta, then checking both render modes before submitting:

```bash
cp scripts/slurm/configs/l1_vggt.yaml scripts/slurm/configs/<name>.yaml
bash scripts/slurm/launch.sh <name> --dry-run
bash scripts/slurm/launch.sh <name> --smoke --dry-run
```

See [`scripts/slurm/README.md`](scripts/slurm/README.md) for the full config
schema, smoke/prod mode semantics, monitoring commands, and launcher tests.


## Watch HTML inside the Cluster
From the repo on the cluster (you're presumably on Remote-SSH), run in the VS Code terminal:
    cd docs && python3 -m http.server 8000
