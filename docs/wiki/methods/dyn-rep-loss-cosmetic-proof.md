---
title: "loss/dyn ≡ loss/rep is cosmetic — synthetic JAX proof"
tags: [r2dreamer, kl-loss, code-audit]
date: 2026-04-27
status: verified
---

# `loss/dyn` ≡ `loss/rep` ≡ `latent/kl_divergence` — cosmetic, no fix needed

## Question

A smoke audit of `modules/r2dreamer/agent.py` flagged that the three logged
KL metrics `loss/dyn`, `loss/rep`, and `latent/kl_divergence` are bit-equal
in the metrics CSV across every recent run. Is this a bug (broken
bookkeeping that hides one of the losses) or a property of how
`stop_gradient` interacts with logging?

## Code under test

`modules/r2dreamer/agent.py:404-409` — both losses are scalar means of the
two arrays returned by `_kl_loss`:

```python
dyn_loss, rep_loss = _kl_loss(
    post_logits_flat, prior_logits_flat,
    cfg.stoch_classes, cfg.stoch_discrete, cfg.kl_free
)
losses["dyn"] = jnp.mean(dyn_loss)
losses["rep"] = jnp.mean(rep_loss)
```

`agent.py:621-622, 631` — what gets logged:

```python
for k, v in losses.items():
    metrics[f"loss/{k}"] = v
...
metrics["latent/kl_divergence"] = jnp.mean(dyn_loss)
```

`agent.py:645-681` — the loss core:

```python
def _kl_loss(post_logits, prior_logits, stoch_classes, stoch_discrete, kl_free):
    post_probs = jax.nn.softmax(post_logits, axis=-1)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)
    post_log = jnp.log(post_probs + 1e-8)
    prior_log = jnp.log(prior_probs + 1e-8)

    def _kl(p, logp, logq):
        return jnp.sum(p * (logp - logq), axis=-1)

    # dyn: train prior toward frozen posterior
    sg_post_probs = jax.lax.stop_gradient(post_probs)
    sg_post_log   = jax.lax.stop_gradient(post_log)
    kl_dyn   = jnp.sum(_kl(sg_post_probs, sg_post_log, prior_log), axis=-1)
    dyn_loss = jnp.maximum(kl_dyn, kl_free)

    # rep: train posterior toward frozen prior
    sg_prior_log = jax.lax.stop_gradient(prior_log)
    kl_rep   = jnp.sum(_kl(post_probs, post_log, sg_prior_log), axis=-1)
    rep_loss = jnp.maximum(kl_rep, kl_free)
    return dyn_loss, rep_loss
```

## Why the forward values must coincide

`jax.lax.stop_gradient` is the identity in the forward pass. Therefore

  `KL(sg(post) ‖ prior) ≡ KL(post ‖ prior) ≡ KL(post ‖ sg(prior))`

as numerical arrays. The clipping `maximum(·, kl_free)` is applied to the
same elementwise array on both sides, so element-by-element equality is
preserved. The two metrics differ only in **where** the gradient is
allowed to flow:

| metric        | forward value   | grad reaches `post`     | grad reaches `prior` |
| ------------- | --------------- | ----------------------- | -------------------- |
| `loss/dyn`    | `KL(post‖prior)`| no (sg)                 | yes                  |
| `loss/rep`    | `KL(post‖prior)`| yes                     | no (sg)              |
| `latent/kl_…` | `mean(dyn)`     | logging-only, no `grad` | n/a                  |

The training scales `cfg.scale_dyn = 1.0` and `cfg.scale_rep = 0.1`
(`config.py`) are applied inside `total_loss` only — the metrics dict
records the raw, pre-scale values. So the three numbers are by
construction bit-identical *and* the optimizer still sees the asymmetric
gradient signal. No fix needed.

## Synthetic proof

Self-contained JAX script (`/tmp/verify_dyn_rep_symmetry.py`). Mirrors
`_kl_loss` exactly, builds `(B=4, T=8, C=32, K=16)` logits matching the
agent's latent shape, then checks (i) forward equality, (ii) gradient
zero/nonzero structure, (iii) scale routing inside `total_loss`.

```python
"""Synthetic proof that loss/dyn ≡ loss/rep is cosmetic.

Mirrors `_kl_loss` in modules/r2dreamer/agent.py:645-681.
"""

import jax
import jax.numpy as jnp


# ---- replica of agent._kl_loss (lines 645-681) -------------------------------
def _kl_loss(post_logits, prior_logits, kl_free):
    post_probs = jax.nn.softmax(post_logits, axis=-1)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)

    post_log = jnp.log(post_probs + 1e-8)
    prior_log = jnp.log(prior_probs + 1e-8)

    def _kl(p, logp, logq):
        return jnp.sum(p * (logp - logq), axis=-1)

    # vanilla KL(post || prior) — no stop_gradient
    kl_post_prior = _kl(post_probs, post_log, prior_log)
    kl_vanilla = jnp.sum(kl_post_prior, axis=-1)

    # dyn: train prior toward frozen posterior
    sg_post_probs = jax.lax.stop_gradient(post_probs)
    sg_post_log = jax.lax.stop_gradient(post_log)
    kl_dyn = jnp.sum(_kl(sg_post_probs, sg_post_log, prior_log), axis=-1)
    dyn_loss = jnp.maximum(kl_dyn, kl_free)

    # rep: train posterior toward frozen prior
    sg_prior_log = jax.lax.stop_gradient(prior_log)
    kl_rep = jnp.sum(_kl(post_probs, post_log, sg_prior_log), axis=-1)
    rep_loss = jnp.maximum(kl_rep, kl_free)

    kl_van_clipped = jnp.maximum(kl_vanilla, kl_free)
    return dyn_loss, rep_loss, kl_van_clipped


def dyn_loss_fn(post_logits, prior_logits, kl_free):
    d, _, _ = _kl_loss(post_logits, prior_logits, kl_free)
    return jnp.mean(d)


def rep_loss_fn(post_logits, prior_logits, kl_free):
    _, r, _ = _kl_loss(post_logits, prior_logits, kl_free)
    return jnp.mean(r)


def total_loss_fn(post_logits, prior_logits, kl_free, scale_dyn, scale_rep):
    d, r, _ = _kl_loss(post_logits, prior_logits, kl_free)
    return scale_dyn * jnp.mean(d) + scale_rep * jnp.mean(r)


def main():
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    shape = (4, 8, 32, 16)  # (B, T, stoch_classes, stoch_discrete)
    post_logits  = jax.random.normal(k1, shape) * 3.0
    prior_logits = jax.random.normal(k2, shape) * 3.0

    B, T, C, K = shape
    post_flat  = post_logits.reshape(B * T, C, K)
    prior_flat = prior_logits.reshape(B * T, C, K)
    kl_free, scale_dyn, scale_rep = 1.0, 1.0, 0.1

    # forward
    dyn, rep, vanilla = _kl_loss(post_flat, prior_flat, kl_free)
    dyn_val, rep_val, van_val = float(jnp.mean(dyn)), float(jnp.mean(rep)), float(jnp.mean(vanilla))
    forward_equal = abs(dyn_val - rep_val) < 1e-10 and abs(dyn_val - van_val) < 1e-10

    # backward
    g_dp = jax.grad(dyn_loss_fn, argnums=0)(post_flat, prior_flat, kl_free)
    g_dq = jax.grad(dyn_loss_fn, argnums=1)(post_flat, prior_flat, kl_free)
    g_rp = jax.grad(rep_loss_fn, argnums=0)(post_flat, prior_flat, kl_free)
    g_rq = jax.grad(rep_loss_fn, argnums=1)(post_flat, prior_flat, kl_free)
    norm = lambda x: float(jnp.sqrt(jnp.sum(x * x)))

    # scaling inside total_loss
    g_tp = jax.grad(total_loss_fn, argnums=0)(post_flat, prior_flat, kl_free, scale_dyn, scale_rep)
    g_tq = jax.grad(total_loss_fn, argnums=1)(post_flat, prior_flat, kl_free, scale_dyn, scale_rep)
    diff_post  = float(jnp.max(jnp.abs(g_tp - scale_rep * g_rp)))
    diff_prior = float(jnp.max(jnp.abs(g_tq - scale_dyn * g_dq)))
    # ... (full script also prints a clean report; see /tmp/verify_dyn_rep_symmetry.py)
```

The full script is at `/tmp/verify_dyn_rep_symmetry.py`. Re-run with:

```bash
MAIN_VENV=/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/.venv
$MAIN_VENV/bin/python /tmp/verify_dyn_rep_symmetry.py
```

## Numerical results

```
========================================================================
FORWARD IDENTITY  (stop_gradient is identity in the forward pass)
========================================================================
  loss/dyn                 = 147.5278167725
  loss/rep                 = 147.5278167725
  vanilla KL(post||prior)  = 147.5278167725
  |dyn - rep|              = 0.000e+00
  |dyn - vanilla|          = 0.000e+00
  forward bit-equal?       = True

========================================================================
BACKWARD ASYMMETRY  (stop_gradient blocks the protected branch)
========================================================================
  ||d(dyn)/d(post)||        = 0.000e+00   (expect 0)
  ||d(dyn)/d(prior)||       = 9.099e-01   (expect >0)
  ||d(rep)/d(post)||        = 1.036e+00   (expect >0)
  ||d(rep)/d(prior)||       = 0.000e+00   (expect 0)
  zero/nonzero structure ok = True

========================================================================
SCALING  (scale_dyn=1.0, scale_rep=0.1 apply only inside total_loss)
========================================================================
  max|d(total)/d(post)  - 0.1 * d(rep)/d(post)|   = 5.355e-09
  max|d(total)/d(prior) - 1.0 * d(dyn)/d(prior)|  = 0.000e+00
  ratio ||d(total)/d(post)|| / ||d(rep)/d(post)|| = 0.100000   (expect 0.1)
  ratio ||d(total)/d(prior)|| / ||d(dyn)/d(prior)|| = 1.000000   (expect 1.0)

========================================================================
VERDICT
========================================================================
  PASS — loss/dyn ≡ loss/rep is COSMETIC.
  Forward values coincide by stop_gradient symmetry;
  backward gradients route correctly to disjoint params with
  the configured scales.
```

## Conclusion

The bit-equality of `loss/dyn`, `loss/rep`, and `latent/kl_divergence`
in every metrics CSV is **cosmetic**: forward values coincide by
construction (`stop_gradient` is identity on the forward pass), while
the backward pass still routes a non-zero, scale-correct gradient into
disjoint parameter sets — `prior` for `dyn`, `post` for `rep`. No code
change required; the smoke-audit alarm is closed. (If a future debug
needs to distinguish the two, log `kl_dyn` *before* the `maximum`
clipping, or log a free-nat hit-rate — both signal training health
without breaking the existing metric contract.)

## Critical re-examination — DreamerV3 path comparison

The synthetic JAX proof above is a closed-form argument on the *current*
R2-Dreamer JAX `_kl_loss` shape. It is silent on whether DreamerV3 or
canonical R2-Dreamer (Bansal et al.) implement an *additional* mechanism —
asymmetric `kl_balance`, asymmetric clip placement, asymmetric scheduling,
or a precision-induced drift — that would make `dyn ≠ rep` in forward in
upstream and would therefore expose R2-Dreamer JAX as silently
under-implementing a real DreamerV3 feature. This section closes that gap
by reading the upstream sources directly.

Sources audited:

- `external/dreamerv3-official/dreamerv3/rssm.py:120-133` (Hafner JAX,
  vanilla DreamerV3)
- `external/dreamerv3-official/dreamerv3/agent.py:237-240` (loss
  aggregation + logging)
- `external/dreamerv3-official/dreamerv3/configs.yaml:86`
- `external/dreamerv3-torch/networks.py:272-290` (NM512 PyTorch port)
- `external/dreamerv3-torch/configs.yaml:57-59`
- `external/r2dreamer/rssm.py:222-230` (Bansal et al., the actual
  port-base for `modules/r2dreamer/`)
- `modules/r2dreamer/agent.py:404-409, 609-611, 645-681`,
  `modules/r2dreamer/config.py:60-75`

| # | Question | Verdict | Evidence |
|---|----------|---------|----------|
| Q1 | Does upstream have a `kl_balance` term that breaks forward symmetry between dyn and rep? | **CONFIRMS-COSMETIC** | DreamerV3 (v3) abandoned DreamerV2's `kl_balance`. `dreamerv3-official/rssm.py:125-126` uses `dyn = D(sg(post)).kl(D(prior))`, `rep = D(post).kl(D(sg(prior)))` — same pattern as R2-Dreamer JAX. No `kl_balance` symbol in any of `dreamerv3-official/`, `dreamerv3-torch/`, `external/r2dreamer/`. The asymmetry lives in the **loss scales** (`dyn=1.0, rep=0.1`), not the forward KL. |
| Q2 | Is free-bits clip applied symmetrically (same threshold, same axis) to dyn and rep? | **CONFIRMS-COSMETIC** | DreamerV3-official: `jnp.maximum(dyn, free_nats); jnp.maximum(rep, free_nats)` (`rssm.py:127-129`). DreamerV3-torch: `torch.clip(rep_loss, min=free); torch.clip(dyn_loss, min=free)` (`networks.py:286-287`). Canonical R2-Dreamer: `torch.clip(rep_loss, min=free); torch.clip(dyn_loss, min=free)` (`rssm.py:227-228`). All three use **same threshold, same axis (post-sum), same op** for both branches. R2-Dreamer JAX matches: `jnp.maximum(kl_dyn, kl_free); jnp.maximum(kl_rep, kl_free)` (`agent.py:674,679`). |
| Q3 | Are dyn and rep ever multiplied by different masks/weights/schedules? | **CONFIRMS-COSMETIC** | DreamerV3-official `agent.py:237-240` aggregates with a single `self.scales` dict — no warmup, no per-loss mask, no batch-importance weight on either branch. R2-Dreamer JAX `agent.py:609-611` also uses pure constant scales. No path applies a time-varying or per-sample weight that differs between dyn and rep in any of the four codebases. |
| Q4 | Does upstream log raw or scaled KL? Would matching change the claim? | **CONFIRMS-COSMETIC** | DreamerV3-official `agent.py:239`: `metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})` — logs the **raw, pre-scale** loss. So upstream's `loss/dyn` and `loss/rep` *also* coincide bit-equally in their CSV (just like ours). DreamerV3-torch `models.py:157-158` logs `to_np(torch.mean(dyn_loss))` and `to_np(torch.mean(rep_loss))` — also raw, same numerics. The cosmetic claim therefore reproduces upstream behavior exactly, not a port-drift. |
| Q5 | Does forward bit-equality survive bf16 / fp16? | **CONFIRMS-COSMETIC** | New synthetic check `/tmp/verify_dyn_rep_bf16.py` (log: `/tmp/dyn_rep_bf16.log`) re-runs the symmetry probe in {fp32, bf16, fp16}. Result: `max|d-r| = 0.000e+00` and `sum|d-r| = 0.000e+00` in **all three dtypes**. Reason: `stop_gradient` is the identity in the forward pass even under low-precision, and the two paths share the *same* probability tensor with the *same* reduction order — there is no non-associative add reordering between them, so the output bits are identical regardless of dtype. (Note: the *value* drifts across dtypes — fp32: 98.893509, bf16: 98.906250, fp16: 98.896484 — but dyn vs rep within each dtype is bit-equal.) |
| Q6 | Does upstream apply asymmetric scales separately or via a single kl_scale + kl_balance? | **CONFIRMS-COSMETIC** | `dreamerv3-official/configs.yaml:86`: `loss_scales: {dyn: 1.0, rep: 0.1, ...}`. `dreamerv3-torch/configs.yaml:57-58`: `dyn_scale: 0.5, rep_scale: 0.1`. Both are **separate scalars in the aggregation**, not a single `kl_scale` modulated by `kl_balance`. R2-Dreamer JAX uses the same shape (`scale_dyn=1.0, scale_rep=0.1`, `agent.py:609-611`) and matches the DreamerV3-official numeric values exactly (`dreamerv3-torch` is the outlier with `dyn=0.5`, irrelevant to canonical R2-Dreamer). |

### Synthesis

Both halves of the original claim survive scrutiny:

1. **"Cosmetic" claim — survives.** Across DreamerV3 (Hafner), DreamerV3-torch
   (NM512), canonical R2-Dreamer (Bansal et al.), and our R2-Dreamer JAX, the
   forward KL on the dyn and rep branches is mathematically identical
   (the `stop_gradient` is forward-identity, the free-bits clip is symmetric,
   and there is no `kl_balance` in v3-era code). Logging raw pre-scale
   values, as all four codebases do, *will* produce bit-equal `loss/dyn`
   and `loss/rep` traces in CSV — that is the upstream-correct behavior, not
   a port artefact.

2. **"No impact at 2.4M steps" claim — survives.** No upstream codebase
   carries a missing mechanism that would create a forward-asymmetry between
   dyn and rep. The asymmetric *training signal* lives entirely in the
   per-loss scales (`scale_dyn=1.0`, `scale_rep=0.1`) plus the disjoint
   gradient routing through `stop_gradient`, both of which R2-Dreamer JAX
   already implements (proven by the original synthetic backward check above).
   No optimizer-state interaction is plausible: AdamW's first/second moments
   accumulate per-parameter, and dyn/rep already write to disjoint
   parameter subsets via `stop_gradient`, so there is no shared optimizer
   state through which the equal forward values could leak asymmetric
   behavior.

3. **bf16 numerical-precision check — no change.** All three dtypes
   (fp32, bf16, fp16) produce bit-equal `dyn` vs `rep` in forward. The
   absolute KL value drifts across dtypes (~0.013 nats between fp32 and
   bf16, expected for ~7-bit mantissa truncation on a ~99-nat sum), but
   the dyn/rep equality is preserved within each dtype.

4. **Severity for pending actfix reruns: zero.** The cosmetic claim does
   not affect the three currently-PENDING actfix reruns. There is no
   missing mechanism to gate behind. The reruns can proceed.

5. **Cross-link to canonical R2-Dreamer divergence audit.** The one place
   R2-Dreamer JAX *does* deviate from canonical R2-Dreamer is the kl_free
   per-group floor — see [`kl-free-per-group-fix.md`](kl-free-per-group-fix.md).
   That is a **different** divergence (clip *granularity*, not clip
   *symmetry*); it does not interact with the dyn/rep cosmetic question
   because canonical R2-Dreamer applies its (summed-axis) clip identically
   to both branches, and so does R2-Dreamer JAX.

## Related

- Smoke audit that surfaced the question: `modules/r2dreamer/agent.py:404-409, 621-622, 631, 645-681`.
- Active posterior-collapse fix (separate finding from the same audit, tracked independently — not addressed here).
- Companion divergence audit (different question, different mechanism):
  [`kl-free-per-group-fix.md`](kl-free-per-group-fix.md).
- Numerical-precision script: `/tmp/verify_dyn_rep_bf16.py`, log
  `/tmp/dyn_rep_bf16.log`.
