# A world model can be learned from world points + camera pose, but underperforms RGB on Level 1

**Scope:** Level 1 ObjectNav (1 house, chair-only). Single seed per arm.
**Runs compared (matched task, ~2M+ steps, scalars-only):**

| arm | encoder | observation | run | final step |
|---|---|---|---|---|
| **WP+CP (flatten)** | `vggt` | 37²×3 world points + 9-D camera pose (4116-D), flattened → linear readout | `vggt_jax-4216462` | 2.14M |
| **RGB baseline** | `cnn` | 3×64×64 RGB image | `4194043` (`lhgoxh0y`) | 2.40M |

Both arms share the same Dreamer world-model backbone and training
hyperparameters; they differ only in the observation modality and its encoder
front-end.

## A world model *is* learnable from geometry alone

The central question is whether the agent can learn a predictive world model
when its only observation is VGGT's pooled world-point map plus the camera
pose — i.e. with the raw image removed entirely. The latent-entropy dynamics
answer this in the affirmative. Throughout training the **prior entropy tracks
the posterior entropy** at a small, stable offset (≈ +0.19 nats), and this gap
is essentially identical to the RGB baseline's (≈ +0.18 nats):

| | posterior H | prior H | prior−posterior gap |
|---|---|---|---|
| WP+CP @ 107k → 2.14M | 1.25 → 1.48 | 1.42 → 1.67 | +0.17 → +0.19 |
| RGB   @ 120k → 2.40M | 0.35 → 0.66 | 0.47 → 0.86 | +0.12 → +0.20 |

Because the prior is produced by the dynamics model from the previous latent
*without* seeing the current observation, a prior that stays this close to the
observation-informed posterior means the dynamics predictor has learned to
anticipate the next latent state from the WP+CP+pose stream. The representation
does not collapse (entropy stays well above zero) and the dynamics KL grows
smoothly (4.5 → 6.3), the expected signature of a world model that is acquiring
structure rather than degenerating. The policy follows: success rate rises from
9% to ~65% and episode reward becomes solidly positive (≈ +5.6). **A Dreamer
world model can therefore be trained from world points and camera pose alone, with
no RGB input.**

## …but it is a weaker representation than raw images at Level 1

![Level-1 success rate: WP+CP flatten vs RGB CNN baseline. The RGB agent learns
markedly faster (≈70% by 0.5M steps) while the WP+CP agent lags through the
first ~1M steps and converges just below the baseline. Faint lines are raw
per-log SR; bold lines are a rolling mean; end labels are the final 200-point
average.](../images/l1-wpcp-vs-cnn-sr.png)

*Figure 1: Success rate vs environment steps on Level 1 ObjectNav. WP+CP
(flatten) reaches ~65%, the RGB CNN baseline ~71%; the gap is largest early
(9% vs 36% at 100k) and narrows but never closes.*

While learnable, the geometric observation is a worse learning signal than the
raw image on this task, on every axis measured:

| metric (final, 200-pt avg) | WP+CP flatten | RGB baseline | Δ |
|---|---|---|---|
| **Success rate (SR)** | **65%** | **71%** | −6 pp |
| **SPL** (path-efficiency-weighted) | 44% | 52% | −8 pp |
| Episode reward | +5.6 | +6.9 | −1.3 |
| SR peak | 81% | 100% | — |

The gap is widest in **sample efficiency**. At 100k steps the RGB agent already
reaches 36% SR while the WP+CP agent is at 9%; at 500k the split is 71% vs 38%.
The geometric arm only narrows the gap late in training (both ≈ 60% by 1–2M),
and never overtakes the baseline. The **SPL deficit (−8 pp) exceeds the SR
deficit (−6 pp)**: even on episodes it solves, the WP+CP agent takes less
direct routes, indicating a coarser spatial signal for fine path selection.

This underperformance is consistent with the latent-entropy levels. The WP+CP
world model operates at a markedly **higher absolute posterior entropy (≈ 1.48
vs 0.66 nats)** than the RGB model — its belief state stays less compressed and
more uncertain. The 37²-pooled world-point map discards the per-pixel texture
and edge cues that a CNN exploits, leaving a representation that the dynamics
model fits just as *tightly* (equal prior-tracking gap) but that is
intrinsically *less informative* for this single-scene, single-goal task.

## Takeaway

Level 1 establishes a proof of concept with a clear cost: removing the image
and handing Dreamer only world points + camera pose still yields a functioning
world model and a competent navigator (~65% SR), but at a 6 pp SR / 8 pp SPL
penalty and substantially slower learning than the RGB baseline. Whether the
geometric representation's advantages emerge on harder curriculum levels
(multi-goal L2, multi-house L3/L4, where 3D structure should transfer across
scenes better than scene-specific RGB textures) is the question the L2–L4
flatten runs are launched to answer.

---
*Source: `output/r2dreamer-curriculum-l1-vggt/run-4216462/metrics.csv`,
`output/r2dreamer-curriculum-l1/run-4194043/metrics.csv`. Single seed per arm;
treat point differences with care.*
