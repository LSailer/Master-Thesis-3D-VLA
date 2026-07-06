# MLP width & capacity: research for the 50/50 house-encoder split

Context: `src/prototyp/house_encoder_capacity/` — `HousePointsCameraEncoder`
(`src/r2dreamer/encoders/mlp.py:103`) splits a 2048-d output 50/50 into a
camera branch (9 DoF → 1024) and a house branch (~1.57M point values → 1024).
This note collects external best-practice on **how wide an MLP can usefully be**
and **how DreamerV3 sizes its MLPs**, then applies it to the split question.

Sources are inline links; see end for full list.

---

## 1. How DreamerV3 sizes its MLPs / RSSM (the downstream consumer)

The encoder output feeds the RSSM posterior, so DreamerV3's own sizing is the
relevant reference point. From the official `config.yaml` (confirmed by the
author in [Issue #131](https://github.com/danijar/dreamerv3/issues/131); the
paper's Table B.1 had a typo, code is authoritative):

| Size | GRU `deter` | MLP `hidden` | `classes` | RSSM MLP layers |
|---|---|---|---|---|
| 12M  | 2048 | 256  | 16 | 1 |
| 25M  | 3072 | 384  | 24 | 2 |
| 50M  | 4096 | 512  | 32 | 3 |
| 100M | 6144 | 768  | 48 | 4 |
| 200M | 8192 | 1024 | 64 | 5 |

Patterns that matter here:

- **`hidden` is modest** — 256 to 1024 — even at 200M params. The bulk of
  DreamerV3's parameters live in the **GRU recurrence (`deter`)**, held at
  ~8× `hidden`. Wide MLPs are *not* how DreamerV3 buys capacity; recurrence and
  the categorical stochastic state are.
- **MLP depth scales with size** (1→5 layers), width scales ~1.5× per step.
  The ETH Zürich scaling study below confirms this is the right lever: for a
  fixed parameter budget, *depth buys more than width* on structured tasks.
- Activation is **LayerNorm + SiLU** — exactly the `RMSNorm + SiLU` block used
  in this repo's encoders (`mlp.py:60-63`). So the block shape already matches
  DreamerV3; the question is purely the *widths*.

**Implication:** the encoder's job is to produce a 2048-vector the RSSM can
ingest. DreamerV3 never feeds an MLP a 1024-wide *raw* feature and asks for a
1024-wide output from a near-empty input — it projects *rich* features
(images, latents) down. A 9→1024 raw Dense is not a pattern DreamerV3 uses.

## 2. General MLP width / capacity best practice

### Width-vs-depth (fixed parameter budget)
[Learnixo](https://learnixo.io/courses/ai-interview-deep-learning/dl-depth-vs-width):
depth enables hierarchical abstraction and gives exponentially more function
classes per log-linear parameter increase; width raises *per-layer* capacity.
For a fixed budget, **deeper-narrower usually beats wider-shallower** on
structured data. Practical defaults:

| Task | Depth | First hidden width |
|---|---|---|
| Tabular / low-DoF | 2–3 | 2–4× input dim, halve each layer (pyramid) |
| Vision w/ skips | 6–12 | bottleneck: expand 4× then compress |

### The capacity / information bottleneck
[DL Notes](https://deeplearningnotes.com/cnns/basics/mlp-limitations): a Dense
layer `d_in → d_out` has `d_in × d_out` params and **cannot represent more
independent output directions than `min(d_in, d_out)`** — the weight matrix has
rank ≤ `d_in`. So:

> **A projection `Dense(d_in → d_out)` with `d_out > d_in` adds parameters but
> no new representational directions.** Width beyond the input rank is wasted
> capacity unless the *input* dimension is the bottleneck you are deliberately
> matching to a downstream consumer (e.g. RSSM expects 2048).

This is the single most load-bearing fact for the split question — see §4.

### Inverted bottleneck (modern MLP best practice)
[Scaling MLPs (ETH Zürich, NeurIPS 2023)](https://arxiv.org/html/2306.13575):
the minimal stable-trainable MLP block is
`Block(z) = z + W_c · σ(W_e · LN(z))` with `W_e` **expanding** ~4× and `W_c`
compressing back, plus pre-LN and skip connections. Wide-then-narrow, not
narrow-then-wide. Combined with GELU/SiLU and skip connections, this is what
makes MLPs scale without optimization instability.

### Capacity is non-monotonic — bigger is not always better
[Beyond scaling (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S092523122601564X):
enlarging width/depth does **not** guarantee monotonic improvement. Two failure
modes — (1) approximation floors for fixed-size ReLU nets, (2) stability
breaches where larger spectral scales push a fixed step size out of the stable
regime → *higher* training loss for the bigger model. **Co-tune capacity with
optimizer stability; moderate capacity is often optimal.**

### Low-DOF inputs: expand via features, not raw Dense
Pose/coordinate literature ([MotionMixer](https://doi.org/10.24963/ijcai.2022/111),
[siMLPE](https://doi.org/10.1109/wacv56688.2023.00479),
[Learnable Fourier Features](https://arxiv.org/pdf/2106.02795)): raw
low-dimensional inputs (9 DoF, 3×J joints) are projected to ~32–128 dims, often
through **Fourier/positional features** rather than a plain Dense, because MLPs
have spectral bias toward low-frequency functions (Tancik 2020). A 9→1024 *raw*
Dense is far outside the practiced range; 9→64 with Fourier features then to
the fused width would be the idiomatic shape.

## 3. Parameter-count intuition for this encoder

Per-branch parameter counts for the current defaults
(`embed_dim=1024`, `camera_hidden=1024`, `camera_layers=1`,
`point_hidden=256`, `point_layers=2`):

| Branch | Dominant matrices | ≈ params |
|---|---|---|
| Camera | `Dense(9→1024)` + `Dense(1024→1024)` | ~1.05M |
| House per-point | `Dense(6→256)` + `Dense(256→256)` | ~0.07M (**shared across all N points**) |
| House pool→proj | `Dense(512→1024)` | ~0.52M |

The house per-point MLP is tiny (≈70k params) but runs **N times per step**; the
house *projection* (`512→1024`, 0.52M) is where width lives. The camera branch
spends ~1.05M params to encode 9 DoF — clearly over-provisioned.

## 4. Applying it to the 50/50 split — the pooling bottleneck dominates

This is the decisive point and it is *not* visible in the HANDOFF table.

Trace the house branch (`mlp.py:134-167`):

```
house_points (N, 6)
  → per-point MLP: Dense(6→256) → RMSNorm/SiLU → Dense(256→256)   # (N, 256)
  → masked mean ‖ masked max                                      # (512,)
  → house_proj Dense(512 → embed_dim)                             # (embed_dim,)
```

The **input to `house_proj` is a 512-vector** (mean‖max of 256-dim point
features). By the rank argument in §2, `Dense(512 → 1984)` can produce at most
**512 independent output directions** no matter how wide the output is. The
extra 962 dims of a 64/1984 reallocation are *interpolated copies* of the same
512-dim pooled signal — capacity is wasted, exactly the "width beyond input
rank is wasted" rule.

So the E2 reallocation (64/1984) in the HANDOFF is **rank-bounded and should
not be expected to move metrics** — for a reason orthogonal to the HANDOFF's
"house half is underused" hypothesis. The binding width is:

1. **`point_hidden = 256`** — the per-point feature width before pooling. This
   sets the rank of everything downstream. Widening *this* (e.g. 256→512) is
   the only way to give the house branch more independent directions, and it
   costs N× per step (bench first — see `house-context-full-buffer-options.md`
   for the dense-cost warning).
2. **The pooling itself** — mean+max is a 2× summary of N points. Mean+max is
   a hard information ceiling regardless of `point_hidden`; this is exactly
   why `house-context-full-buffer-options.md` lists PointNet++ set-abstraction,
   voxel grids (option 4), and Perceiver cross-attention (option 5) as the
   real levers — they replace "2× summary" with "learned, multi-scale summary."

**Recommendation:** flip the HANDOFF decision rule.

- The E2 width-only reallocation (64/1984) tests the wrong lever — it widens
  a projection of a rank-512 input. Run E1 (branch-utilization) first; if the
  house half is saturated *at the current 1024*, the cause is the 512-dim
  pooled input, not the 1024 output.
- If you want to test *width* properly, widen `point_hidden` (256→512) at
  constant `embed_dim`, **not** the output split. That actually raises the
  rank of the house signal.
- The structurally correct fix remains the pooling redesign (PointNet++ /
  option 4 / option 5), which is already scoped out of this folder. The
  width-split question may be closable as "width is not the constraint; the
  pooling summary is" without running E2 at all.

## Sources

- [DreamerV3 paper (arXiv 2301.04104)](https://export.arxiv.org/pdf/2301.04104v1.pdf)
- [danijar/dreamerv3 — config & Issue #131 (model sizes)](https://github.com/danijar/dreamerv3/issues/131)
- [World Model (RSSM) — DeepWiki](https://deepwiki.com/danijar/dreamerv3/4.1-world-model-(rssm))
- [Scaling MLPs: A Tale of Inductive Bias (ETH Zürich, NeurIPS 2023)](https://arxiv.org/html/2306.13575)
- [Beyond scaling: non-monotonic effects of depth and width (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S092523122601564X)
- [MLP limitations / the VRAM wall (DL Notes)](https://deeplearningnotes.com/cnns/basics/mlp-limitations)
- [Network depth vs width (Learnixo)](https://learnixo.io/courses/ai-interview-deep-learning/dl-depth-vs-width)
- [MotionMixer (IJCAI 2022)](https://doi.org/10.24963/ijcai.2022/111)
- [siMLPE: Back to MLP (WACV 2023)](https://doi.org/10.1109/wacv56688.2023.00479)
- [Learnable Fourier Features (arXiv 2106.02795)](https://arxiv.org/pdf/2106.02795)