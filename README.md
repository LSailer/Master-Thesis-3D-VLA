# Master Thesis: VLA + 3D Semantic Scene Understanding

Augmenting Vision-Language-Action models with UNITE 3D features for object navigation in HM3D (Habitat).

## Slides

https://LSailer.github.io/Master-Thesis-3D-VLA/

## Ralph (Autonomous TDD Agent)

```bash
# Start Ralph (picks up to N ready AFK issues, implements via TDD, creates PRs)
gh workflow run ralph.yml -f max_tasks=5 -f time_limit=120

# Check run status
gh run list --workflow=ralph.yml

# See which issues Ralph is working on / finished
gh issue list --label in-progress
gh issue list --label in-review

# See ready tasks waiting for Ralph
gh issue list --label AFK --label ready

# Promote backlog issues to ready (manual)
gh issue edit <N> --remove-label backlog --add-label ready
```
**Partitions:**
| Partition | Use Case | Max Time |
|-----------|----------|----------|
| `dev_gpu_h100` | Testing, validation, quick experiments | 30 min |
| `gpu_h100` | Standard GPU jobs, training | 48h |

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


## GPU + tmux + Remote Control Workflow

Use `start_gpu_session.sh` to run experiments on a GPU node inside tmux, then optionally hand off to Claude remote control.

**1. Start GPU session (tmux wraps srun):**
```bash
./start_gpu_session.sh                          # defaults: gpu-work, gpu_h100, 24h
./start_gpu_session.sh my-exp dev_gpu_h100 00:30:00  # custom name/partition/time
```

**2. Run experiments** on the GPU node.

**3. (Optional) Start Claude with remote control:**
```bash
claude --remote-control "GPU Experiments"
```
Opens a URL + QR code — access from phone/browser at `claude.ai/code`.

**4. Detach tmux** (GPU session keeps running):
```
Ctrl+b  d
```

**5. Reattach later:**
```bash
tmux attach -t gpu-work
```

**Key:** tmux must wrap srun (not the reverse) — detaching tmux preserves the GPU allocation.