# Hybrid Encoder — CNN(RGB) + gated MLP(WP/CP) — 2026-06-01

Implements the hybrid Dreamer input agreed in the 2026-05-28 meeting: feed the
RSSM posterior **both** modalities instead of VGGT-only.

## What it is

```
embed = concat([ CNN(RGB_64) , gate · MLP(WP+CP 4116) ])
```

The concatenated embedding feeds the RSSM posterior. The CNN path is the
standard Dreamer encoder on a 64×64 RGB image; the MLP path consumes the
VGGT world-points + camera-pose features (`world_points` + `camera_pose`,
4116-d).

## Zero-init scalar gate (Flamingo-style)

A single scalar `gate` multiplies the VGGT branch and is **initialised at 0**.
At step 0 the model is exactly CNN-Dreamer; the VGGT branch opens from zero as
training pushes the gate up. We gate the branch (not zero-init the MLP weights)
to avoid a *dead gate*: zeroing both the gate and the MLP gives no gradient
signal to either, so the VGGT path never recovers. Zero-init the gate only and
the MLP stays randomly initialised, so a non-zero gradient flows the moment the
gate lifts off zero.

## Buffer layout

Replay observation is `[rgb_norm 12288 | wp_cp 4116] = 16404` float32. The RGB
slice is the normalised 64×64×3 image (12288); the WP/CP slice is the 4116-d
VGGT feature. Habitat renders at 518 (`render_resolution: 518`, needed by VGGT);
the RGB is downsampled to 64 for the CNN.

## Contribution debug metrics

Logged every `log_every`:

- `hybrid/gate` — current scalar gate value (starts ~0, grows).
- `hybrid/cnn_l2`, `hybrid/vggt_l2` — L2 norm of each branch's contribution.
- `hybrid/cnn_frac`, `hybrid/vggt_frac` — fraction of the fused embedding norm
  from each branch. Read `vggt_frac` as the signal: it starts ~0 and should
  grow if VGGT is being used; if it stays flat at 0 the VGGT path is dead.

## Co-trained decoder (`--decoder`, default off)

Behind `--decoder`, a decoder is co-trained to reconstruct the observation so we
can visually verify the latent. Adds `loss/decoder` to the loss log and
`decoder/reconstructions` image panels in W&B.

## Latent ablation (`--latent_preset {small,default,large}`)

Vary the RSSM latent capacity to see how it interacts with the extra modality.

## Running the smoke

```
# wrapper (submits to gpu_h100_short):
scripts/slurm/launch.sh hybrid_v1 --smoke
# or render only:
python scripts/slurm/launch.py hybrid_v1 --mode smoke --dry-run  # via launch.sh; launch.py: launch.py hybrid_v1 --mode smoke
```

Smoke: 800 steps / 200 prefill, `metrics.csv` ≥ 5 rows (inherited from `_base`).
Confirm VGGT is logged:

```
grep -c "hybrid/vggt_frac" <run>/metrics.csv
```

## Linear mapping

- **3D-50** — gate + small/large latent ablation.
- **3D-51** — co-trained decoder + Linear→MLP projection inside Dreamer.
- **3D-52** — MLP encoder (Dreamer natively uses an MLP encoder).

Baseline to beat: W&B run
`sailer-luca-university-ulm/3d-vla-objectnav/lhgoxh0y` (WP/CP-only).
