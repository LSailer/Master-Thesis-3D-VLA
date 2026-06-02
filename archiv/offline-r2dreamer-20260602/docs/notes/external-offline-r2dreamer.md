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

## 3D-46 — the 3 baseline runs + comparison

### Held-out metrics (built into the training script)

After training, the script computes held-out WM metrics on the last 10% of
episodes and writes `heldout_final.json` + `heldout_table_row.json` to the
output dir (and logs `final/heldout/*` to W&B). Definitions mirror the JAX
`src/r2dreamer/heldout_eval.py` exactly:

- **dynamics_kl / representation_kl** — `rssm.kl_loss(post, prior, kl_free)`
  means (same free-bits clipping as training `loss/dyn` / `loss/rep`). The two
  share a forward value — the `.detach()` only changes which side gets gradient.
- **reward_mse** — MSE of the twohot reward head's expected value
  (`reward(feat).mode()`, real reward space) vs the GT reward.
- **k_step_rollout_mse[k]**, k ∈ {1,5,15} — from the posterior at t=0, run
  `rssm.img_step` with GT actions for k steps, MSE on `get_feat` vs the GT
  posterior feature at step k. (Needs `seq_len ≥ 16`.)
- **reconstruction_nll = NaN** — decoder-free on both sides.

Disable with `--skip-heldout-eval`; tune batches with `--heldout-eval-batches`
(default 64, matching the JAX final eval).

### Launching the runs (SLURM)

```
# smoke (gpu_h100_short, 200 steps, validates GPU + buffer + eval + W&B)
scripts/r2dreamer/slurm/submit_external_offline.sh smoke 0

# the 3 prod runs (gpu_h100_il, 500k steps, ~5 h each, parallelizable)
for s in 0 1 2; do scripts/r2dreamer/slurm/submit_external_offline.sh prod "$s"; done
```

Only `--seed` differs across the 3 prod runs (verifiable from each
`run_config.json`). Throughput ~29 steps/s on one H100 ⇒ ~5 h/seed.

### Comparison table (after the runs finish)

```
python scripts/r2dreamer/build_offline_comparison.py \
    --external-glob 'output/3d46-external-offline/wp_cp-seed*/run-*/heldout_table_row.json' \
    --jax-glob      'output/3d26-offline-ablation/wp_cp-seed*/run-*/heldout_table_row.json' \
    --out-md  docs/notes/offline-ablation-comparison.md \
    --out-csv docs/notes/offline-ablation-comparison.csv
```

If the JAX 3D-26 output dirs aren't on disk, pass `--jax-wandb` instead — it
reads each JAX run's `final/heldout/*` summary from the
`3d-vla-objectnav-offline-ablation` W&B project (tags `3d-26,wp_cp`). The
builder emits a markdown table (JAX vs PyTorch, mean ± std over 3 seeds) and a
per-seed CSV; **reconstruction NLL is excluded** from the head-to-head (N/A on
both sides).

## Tests

`tests/r2dreamer/test_external_offline_buffer.py` validates the split, window
sampling, `is_first` done-shift, one-hot actions, and zeroed initial latent on a
synthetic buffer. Skipped under the main venv (no tensordict); runs under the
external venv.
