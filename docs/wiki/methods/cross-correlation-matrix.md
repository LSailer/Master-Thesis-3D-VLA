# Cross-Correlation Matrix (Barlow Twins form)

A **cross-correlation matrix** `c ∈ ℝ^{D×D}` measures the per-dimension correlation between two D-dimensional feature vectors `x1` and `x2` from the same data point, both batch-normalized (zero mean, unit variance per dim along the batch axis):

```
c[i, j] = (x1ᵀ x2)[i, j] / (B · T)   with x1, x2 normalized per dim
```

- **`c[i, i]` (diagonal)** — Pearson correlation between dim *i* of `x1` and dim *i* of `x2`. Target: **1** ("dim *i* encodes the same information in both views").
- **`c[i, j]` for `i ≠ j` (off-diagonal)** — correlation between *different* dimensions. Target: **0** ("no redundancy across dimensions").

Together the targets define the Barlow Twins criterion: each feature dimension is **invariant** across views and **decorrelated** from every other dimension.

## Use in this codebase

This is *not* used as a contrastive loss between augmented views. In `modules/r2dreamer/agent.py:418–433`, `x1 = projector(rssm_feat)` and `x2 = stop_gradient(VGGT_embed)` — so the cross-correlation matrix regularizes the **RSSM-projected feature to align dimension-wise with the (frozen) VGGT 3D encoder embedding**. It functions as a distillation regularizer, not as self-supervised representation learning.
