# External (PyTorch) R2Dreamer — offline adapter

Adapter that trains the **original PyTorch R2Dreamer** (`external/r2dreamer/`)
offline from the canonical replay buffer, to produce a baseline that is
apples-to-apples with the JAX offline numbers.

- Linear: **3D-45** (this adapter) → **3D-46** (the 3 baseline runs + comparison).
- Parent ablation: **3D-24** (VGGT WP/CP vs Aggregator MLP as R2Dreamer input).
- Entry point: `scripts/r2dreamer/train_external_offline.py`.

## Why an adapter

The stock external entry point (`external/r2dreamer/train.py`) is **env-driven**
(`make_envs` → `OnlineTrainer`, torchrl `LazyTensorStorage` filled by live
rollouts) and logs only to TensorBoard/jsonl. To replay the fixed offline buffer
on a precomputed vector and land in the same W&B project as the JAX runs, the
adapter adds three things without touching the vendored repo:

1. **Vector input** — declares an obs space `{"vector": Box(4116,)}` and sets
   `encoder.mlp_keys='vector'`, `cnn_keys='$^'`. `MultiEncoder` already routes a
   `len(shape)==1` obs matched by `mlp_keys` to its MLP branch; `$^` is the
   match-nothing regex that disables the CNN. No new modeling.
2. **Offline replay** (`OfflineVectorBuffer`) — drop-in for the external
   `Buffer`'s `sample()/update()/count()` interface, so `agent.update()` is
   reused verbatim (identical LaProp/GradScaler/AGC/scheduler behaviour).
3. **W&B** — a thin `tools.Logger` subclass that mirrors scalars to W&B, gated
   behind `--wandb`.

## Fairness: identical to the JAX `OfflineBufferDataset`

The buffer reproduces the JAX sampling contract exactly, so the external sees the
same `(obs, action, reward, flags)` windows the JAX runs saw:

- contiguous windows that **may straddle episode boundaries**;
- `is_first[:, 0] = True`, and `is_first[:, t] = done[t-1]` (reset on `done`);
- `is_terminal == is_last == done` (the buffer records only env `done`);
- actions aligned with obs at the same index (no torchrl one-step shift), one-hot
  over the 4 Habitat actions;
- **zeroed initial latent each batch** — episode reset is driven entirely by
  `is_first`. The external R2-Dreamer replay-buffer latent write-back
  (`replay_buffer.update`) is a **no-op** offline, matching the JAX path which
  carries no cross-batch latent state.

Train/heldout split = `collection_metadata.json["heldout_split"]`
(last 10% of episodes by `episode_id`), same as 3D-25/3D-26.

## Decoder-free on both sides

Default `rep_loss="r2dreamer"` (Barlow-Twins redundancy reduction) carries **no
decoder** — same as the decoder-free JAX R2Dreamer — so there is **no
reconstruction NLL on either side**. Comparable held-out metrics are dynamics KL,
representation KL, reward MSE, and k-step latent rollout error (k ∈ {1,5,15}).
(`rep_loss="dreamer"` would add a `MultiDecoder` and a recon NLL; we do **not**
use it for the baseline.)

## Environment

Run under the **external** venv — it has torchrl/tensordict/gymnasium; the main
`.venv` does not:

```
external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py ...
```

## Commands

Smoke (NaN-free, no W&B; CPU is fine — `torch.compile` is auto-disabled and
`GradScaler` auto-disables without CUDA):

```
external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py \
    --encoder wp_cp --seed 0 --steps 100 \
    --buffer-dir data/offline_buffer_smoke --output-dir /tmp/ext_smoke \
    --device cpu --batch-size 4 --seq-len 16
```

Real run (GPU) — one per seed for 3D-46:

```
external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py \
    --encoder wp_cp --seed 0 --steps 500000 \
    --buffer-dir data/offline_buffer --output-dir output/3d46/ext-wp_cp-seed0 \
    --device cuda:0 --wandb --wandb-name ext-wp_cp-seed0
```

W&B (when `--wandb`): project `3d-vla-objectnav-offline-ablation`, tags
`offline-ablation, 3d-24, framework:pytorch-external, variant:wp_cp`, run name
`ext-<encoder>-seed<n>` — so the runs filter beside the JAX `wp_cp-seed{0,1,2}`.

`run_config.json` (written to `--output-dir`) records both repo and
`external/r2dreamer` code SHAs plus the buffer metadata for reproducibility.

## Handoff to 3D-46

- Launch `--seed {0,1,2}`, `--steps 500000`, `--device cuda:0`, `--wandb`. Runs
  are independent → parallelizable across nodes (mirror the SLURM pattern in
  `scripts/slurm/`).
- Held-out metrics: reuse the train/heldout split above; report dynamics KL,
  representation KL, reward MSE, k-step rollout error (k ∈ {1,5,15}) — **not**
  reconstruction NLL (decoder-free on both sides).
- The only field that may differ across the 3 runs is `--seed`.

## Tests

`tests/r2dreamer/test_external_offline_buffer.py` validates the split, window
sampling, `is_first` done-shift, one-hot actions, and zeroed initial latent on a
synthetic buffer. Skipped under the main venv (no tensordict); runs under the
external venv.
