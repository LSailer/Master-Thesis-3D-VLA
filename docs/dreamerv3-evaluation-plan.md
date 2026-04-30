# DreamerV3 Evaluation Plan

## 1. Smoke Test Notebook

**File:** `notebooks/dreamerv3_smoke_test.ipynb`

- **Environment:** Real Habitat (HM3D ObjectNav)
- **Framework:** JAX DreamerV3 (existing `src/dreamerv3/`)
- **Duration:** ~1000 training steps, ~200 prefill steps, 5-15 min total
- **How to run:** Interactively via Jupyter on GPU node
- **What to plot:** All loss curves (reconstruction, KL, reward, actor, critic)
- **Success criterion:** Eyeball — losses should trend downward
- **WandB:**
  - Project: `dreamerv3-objectnav`
  - Name: `smoke-jax-YYYYMMDD`
  - Tags: `["smoke-test", "jax"]`

## 2. JAX vs PyTorch Comparison Notebook

**File:** `notebooks/dreamerv3_jax_vs_pytorch.ipynb`

- **Environment:** Dummy env (same obs shape `(3, 256, 256)`, 4 discrete actions)
- **JAX side:** Existing `src/dreamerv3/`
- **PyTorch side:** `NM512/dreamerv3-torch` (external repo)
- **Metrics:** Wall-clock time per training step + GPU memory usage
- **Can run in parallel with smoke test**
- **WandB:**
  - Project: `dreamerv3-objectnav`
  - Names: `benchmark-jax-YYYYMMDD`, `benchmark-torch-YYYYMMDD`
  - Tags: `["benchmark", "jax"]` / `["benchmark", "pytorch"]`
  - Group: `jax-vs-pytorch-benchmark`

## 3. 24h Full Training Run

**Prerequisite:** Complete step 2, decide on framework.

- **Submission:** `sbatch` script in `scripts/`
- **GPU:** H100 (`gpu_h100` partition)
- **Duration:** 24 hours
- **Framework:** Winner of JAX vs PyTorch comparison
- **WandB:**
  - Project: `dreamerv3-objectnav`
  - Name: `full-{framework}-YYYYMMDD`
  - Tags: `["full-run", "{framework}"]`

## WandB Conventions

- **Project:** `dreamerv3-objectnav` (single project for all runs)
- **Run naming:** `{purpose}-{framework}-{date}`
- **Tags:** Categorize by run type (`smoke-test`, `benchmark`, `full-run`) and framework (`jax`, `pytorch`)
- **Groups:** Cluster related runs (e.g. `jax-vs-pytorch-benchmark`)
