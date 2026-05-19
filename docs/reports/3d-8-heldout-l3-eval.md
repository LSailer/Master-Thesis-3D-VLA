# 3D-8 Held-Out-House L3 Evaluation

## Goal

Evaluate the L3 CNN/R2-Dreamer ObjectNav checkpoint on HM3D houses that are
disjoint from the original `level3_10houses_1goal` curriculum, then rerun the
baseline under the same held-out eval config.

## Held-Out Curriculum

Generate the eval curriculum:

```bash
python scripts/r2dreamer/build_l3_heldout_curriculum.py
```

Default output:

```text
data/curriculum/level3_heldout_10houses_1goal.json
```

The generator selects 10 chair-containing HM3D train-split scenes with the most
chair episodes after excluding the original L3 scenes. It samples 200 eval
episodes per held-out house by default and records the original L3 scenes in
`heldout_from_scenes`.

## Candidate Checkpoint

The strongest available L3 CNN checkpoint discovered locally is:

```text
output/r2dreamer-curriculum-l3/run-4194045/checkpoints/step_002400000.pkl
```

There is also an earlier actfix run:

```text
output/runs/r2dreamer-curriculum-l3-actfix/run-4119016/checkpoints/step_002400000.pkl
```

Choose one explicitly in the eval artifact note; do not mix checkpoint results.

## Eval Command

```bash
python scripts/r2dreamer/eval_habitat.py \
  --checkpoint output/r2dreamer-curriculum-l3/run-4194045/checkpoints/step_002400000.pkl \
  --encoder cnn \
  --episodes 2000 \
  --curriculum_path data/curriculum/level3_heldout_10houses_1goal.json \
  --output_dir output/eval/l3-heldout-cnn-step-2400000 \
  --split train \
  --render_topdown
```

The curriculum filter uses HM3D train-split scene files; the held-out guarantee
comes from disjoint scene IDs, not the Habitat `val` split.

## Baseline Rerun

Use the same held-out curriculum and episode count for the baseline:

```bash
python scripts/r2dreamer/eval_habitat.py \
  --random \
  --encoder cnn \
  --episodes 2000 \
  --curriculum_path data/curriculum/level3_heldout_10houses_1goal.json \
  --output_dir output/eval/l3-heldout-random \
  --split train
```

## Artifact Checklist

- Commit SHA
- Checkpoint path and checkpoint step
- Generated held-out curriculum path
- Held-out scene list
- Eval command
- W&B run URL, if used
- Aggregate SR, SPL, mean episode length
- Per-house SR for model and baseline
- Thesis wording update location
