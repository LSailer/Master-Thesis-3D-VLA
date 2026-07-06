# The thought process that gets to a quality solution

*Context: "how do we turn per-frame VGGT point maps into a bounded,
structure-preserving house representation a Dreamer agent can condition on?"
— but the process generalizes to any representation-design problem in this
thesis. Each step below cites where it already paid off (or would have caught
a mistake) in the 2026-07-02 graph-experiment session.*

## 1. Define "quality" operationally before designing anything

A representation has no intrinsic quality — only quality *for the decision it
feeds*. Here the consumer is the world model/policy, so the ground-truth
metric is agent performance (episode return, world-model loss on L1), and
everything else — chamfer, PSNR, storage MB — is a proxy. Write down the
proxy → end-metric chain explicitly, so you know which proxy failures matter.

*Evidence from today:* mean chamfer coverage said stride sampling beats
contour sampling; the claim under test ("preserve structure") needed a
different proxy (contour-region fidelity), which reversed the conclusion.
Choosing the metric after seeing results is a trap — derive it from the claim.

## 2. Enumerate the hard constraints first; they prune more than preferences do

For this codebase: JAX static shapes under jit; per-step time budget (extract
~64 ms dominates; anything added must be ≪ that); device memory; the replay
augmentation contract; CPU-only login node (dev loop) vs GPU via Slurm.
A design that violates a hard constraint is dead regardless of its elegance —
check constraints *by arithmetic before writing code*.

*Evidence:* dense adjacency (the tutorial's approach) was rejected by one
multiplication (210k² × 4 B ≈ 176 GB); full-cloud eigendecomposition by the
O(N³) bound the book itself states. Both pruned in minutes on paper.

## 3. Decompose into orthogonal axes and attack each with the cheapest falsifying experiment

The monolithic question "does a graph help?" decomposes into: **storage**
(node table vs edges), **selection** (which points enter the encoder budget),
**encoding** (does structure help the network), **lifecycle** (when to
recompute). One cheap, isolated experiment per axis — that is exactly the
exp1–exp4 folder. Cheap means: saved real data (the 210k-point PLY already on
disk) instead of a new rollout; CPU-runnable; minutes not hours.

*Evidence:* the four experiments answered four different questions with four
different verdicts (edges: recompute don't store / selection: contour wins /
attribute coding: modest, park it / encoding: learnable, pursue). A single
combined experiment would have returned one mushy answer.

## 4. Respect the baseline — verify it before trusting any comparison

The incumbent (even-stride resampling) is the thing to beat, at matched
budgets, on the same data. A broken baseline silently poisons every
conclusion drawn against it.

*Evidence:* the first exp2 run showed stride coverage *degrading* with more
points — physically impossible, which flagged the int32 stride overflow in
the production buffer. Without pausing to ask "can the baseline even do
that?", the experiment would have "proven" contour sampling superior for the
wrong reason, and the production bug would have shipped into full-house runs.
Impossible-looking numbers are gifts: chase them before celebrating them.

## 5. Make results trustworthy: adversarial review + regression tests + determinism

Treat your own experiment code as guilty until reviewed. Independent
review angles (line-by-line, removed-behavior, cross-file) found a real
structural bug (bfloat16 duplicates → self-loops) that all 25 green tests
missed, because the test fixtures lacked the real data's pathology. Fix →
regression test with the pathology (duplicated points) → re-run → confirm the
published numbers moved or didn't. Numbers that survive a found-bug re-run
(ours did: 21.67 dB unchanged) are worth far more than numbers never
challenged.

Corollary: seed everything, keep runs deterministic, and record reproduce
commands next to every table (the results note does this) — otherwise you
cannot tell a regression from noise later.

## 6. Promote winners behind flags; keep the A/B alive in the real loop

Prototype in `src/prototyp/` (throwaway), extract reusable pieces to
`src/prototype_helpers/`, and integrate into the live path only behind a
config flag next to the incumbent (contour-vs-stride snapshot; GNN-vs-MLP
encoder in the `pointnet2.py` seam). The flag is what lets step 7 happen.

## 7. Let the end-to-end metric arbitrate

Final gate: run the L1 smoke (`smoke_house_points_pose.sbatch` pattern) with
each variant and compare training curves / returns. Proxies got the candidate
to the door; only the agent decides if it enters. Be prepared for the proxy
and the end metric to disagree — that outcome is a thesis finding, not a
failure.

## 8. Write down the negative results and the why

Block-GFT compression "failed" (modest ratios, not the bottleneck) — recorded
with numbers and the reason (VGGT color isn't graph-smooth; RGB is a third of
an already-small table). A thesis needs the pruned branches with evidence, and
future-you needs them to not re-explore dead ends.

---

**One-line version:** define quality as the downstream decision's metric →
prune by hard constraints on paper → decompose into orthogonal claims, each
with the cheapest falsifying experiment against a *verified* baseline → make
surviving results trustworthy (review, regression tests, determinism) →
promote behind flags → let the end-to-end agent metric make the final call →
document the corpses.
