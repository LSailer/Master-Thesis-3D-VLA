# WP-resolution / encoder prod-run analysis (3D-50/51/52)

**Date:** 2026-06-01 · **Branch:** `lucasailerls/3d-50-hybrid-cnn-vggt`
**Figures:** `output/analysis/wp-cp-resolution/` · **Script:** `scripts/r2dreamer/analyze_wp_resolution_runs.py`

L1 ObjectNav (1 house, 1 goal, chair-only). All runs share the same world-model
backbone and training hyperparameters (`deter 2048`, `stoch 32×16`, `train_ratio 512`,
`lr 4e-5`, `act_entropy 0.03`, `barlow 5e-4`, `prefill 5000`). They differ **only**
in the encoder front-end / observation.

| run | encoder | obs | readout | SHA / branch | last step | status |
|---|---|---|---|---|---|---|
| 4887404 | `vggt` | WP 37²+CP (4116) | **depth-3 MLP** | `2f6dec3` (current) | ~191k | running, walls out ≈1.37M |
| 4888735 | `vggt_wp_cp_64` | WP 64²+CP (12297) | **depth-3 MLP** | `28ff634` (current) | 50k | hung→relaunched (4889382) |
| 4216462 | `vggt` | WP 37²+CP (4116) | linear | `64f3ddd` (older) | 2.14M | reference |
| 4194043 (`lhgoxh0y`) | `cnn` | RGB 3×64² | conv | older | 2.40M | reference (canonical CNN) |

> **Throughput note:** at the measured steady-state 8.04 steps/s, run 4887404 reaches
> only ~1.37M of 2M steps before the 48 h SLURM wall (would need ~62 h for 2M). All
> current runs wall out ≈1.3–1.4M, so final SR is read at a common ~1.3M ceiling.

The controlled pair (4887404 vs 4888735) is identical except WP pool resolution
(37²→64²). The two `[ref]` runs are fully-trained but from **older branches**, so
they are context, not a controlled comparison.

## Matched-step success rate (read at the *same* env step)

| run | SR@50k | SR@185k | SR@end |
|---|---|---|---|
| WP 37² MLP (cur) | 4.0% | **11.0%** | — (still rising, ~191k) |
| WP 64² MLP (cur) | 3.0% | — | — (hung @50k) |
| WP 37² linear [ref] | 11.0% | **12.0%** | 61% @2.14M |
| RGB CNN 64² [ref] (`lhgoxh0y`) | **16.0%** | **49.0%** | **72% @2.4M (peak >75%, SPL 0.52)** |

## Findings

1. **All runs are training healthily.** WM losses (dyn/rep KL rising 3→6–8, reward
   pred ≈0, Barlow collapses after init), actor/critic stable, posterior/prior
   latent entropy tracking (2.75→1.1), encoder-L2 linear, `nan_skipped=0`. No pathology.

2. **Resolution ablation (controlled, current code) — no benefit visible yet.**
   At the only common window (50k), 64² is **1 pp *below* 37²** (3.0% vs 4.0% SR;
   SPL 0.013 vs 0.023). The larger 12,297-dim obs does not help early learning and
   may slightly slow it. **Inconclusive**: 50k is very early, single-seed, and SR
   moves in ~1 pp quanta. Needs the relaunched 64² (4889382) to reach ~185k+.

3. **Depth-3 MLP vs linear readout (37²) — a wash so far.** At matched 185k the
   current MLP (11%) and the linear reference (12%) are tied. The 50k gap (4% vs 11%)
   was early-trajectory variance, not a real MLP penalty.

4. **Encoder-family gap is the dominant effect.** RGB-CNN (`lhgoxh0y`) leads from the
   very start (**16% SR @50k, 49% @185k**) while every VGGT-WP encoder sits at ~3–11%
   in that window. At convergence CNN reaches **72% final / >75% peak (SPL 0.52)** vs
   the WP-linear ref's 61% (SPL 0.42) — a real **~10 pp asymptotic gap on top of the
   large early-learning gap**. Because the WP-*linear* reference (old branch) tracks
   the current WP-*MLP*, this is an encoder-family effect, not merely a code-branch
   artifact: the metric 3D world-point representation is both *slower to learn from*
   and *lower-ceiling* than raw RGB on L1.

## Caveats

- **Single seed per run** (`seed = SLURM_JOB_ID`) → no error bars; treat 1–2 pp gaps as noise.
- **References are older branches** → reward shaping / env details may differ from current code.
- **Current runs are early** (50k–185k of a 2M target) and will **wall out ≈1.3M**
  at the 48 h SLURM limit, so final SR will be read at a common ~1.3M ceiling, not 2M.

## Next

- Let 4887404 (37² MLP) and 4889382 (64² MLP, fresh) mature; re-run this script and
  compare the controlled pair at a common ~185k, then ~1.3M.
- If the WP-slow-vs-CNN gap holds at maturity, that is a reportable result: the
  metric-3D world-point representation is *harder to learn from early* than RGB on L1,
  even though it reaches comparable asymptotic SR.
