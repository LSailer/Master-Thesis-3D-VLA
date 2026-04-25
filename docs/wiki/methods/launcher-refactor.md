# Launcher Refactor — Encoder ABC + Per-Level Shims + Test Pyramid

**Date**: 2026-04-25
**Closes (planned)**: [#85](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/85), [#52](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/52) (subsumed)
**Touch points**: `modules/r2dreamer/launch/` (new), `modules/r2dreamer/adapters/` (new), `modules/r2dreamer/scripts/run_jax_*.py` (rewritten as shims), `modules/r2dreamer/scripts/slurm/*.sbatch` (path edits), `archiv/r2dreamer-pytorch-20260425/` (new)

## Motivation

The three r2dreamer training entrypoints (`run_jax_habitat`, `run_jax_habitat_vggt`, `run_jax_crafter`) each carried ~80 lines of argparse + Trainer wiring that differed only in env/encoder/adapter plumbing. Issue #85 proposed extracting that into a launcher; issue #52 catalogued the broader script-boilerplate problem (~390 lines duplicated across 13 scripts).

A 2026-04-25 design grill (this document) extended #85's original RFC by:

1. Auditing #52's table against the post-archive (`modules/dreamerv3/`) reality and confirming PyTorch scripts are dormant.
2. Replacing the RFC's `Curriculum` dataclass with a `dict[str, Path]` because curriculum JSON files at `data/curriculum/level{1-4}_*.json` are already the source of truth.
3. Replacing the RFC's `(factory, adapter_class)` tuple registry with an `Encoder` ABC + inheritance because CNN (encoder lives in agent) and VGGT (encoder is external) are structurally asymmetric.
4. Replacing the RFC's "either per-level scripts or `--curriculum` flag" punt with a hard call for **per-level shims**, because each sbatch already couples `(encoder, level, output_dir, wandb_name)` and putting that coupling in code prevents wrong-directory drift.
5. Promoting the test fixture from "synthesized fake obs" to "recycled `output/vggt/parity/` arrays" so L3 tests check numerical correctness against the validated parity baseline, not just shapes.

## The 12 design decisions

| # | Branch | Answer | Rationale |
|---|---|---|---|
| 1 | Scope vs #52 | **(c) Full absorption** | "#85 supersedes #52" claim was leaky; redesigned to truly absorb every #52 row reachable. PyTorch rows go via archive instead of refactor. |
| 2 | Surface shape | **(2c) Sibling functions + `parity/` sub-package** | `train()`, `evaluate()` share most args (curriculum, encoder, env). Parity/benchmark take `(jax_encoder, pytorch_encoder)` — fundamentally different signature, belongs in a sub-package. |
| 3 | PyTorch scripts (`run_pytorch_crafter`, `run_pytorch_standalone`) | **→ `archiv/r2dreamer-pytorch-20260425/`** | Audit (this session) showed: not in active sbatch curriculum jobs, not cited as source of any wiki experiment, not modified since 2026-03-29 (creation/move only). |
| 4 | Curriculum representation | **(4c) `dict[str, Path]`** | JSON files at `data/curriculum/level{1-4}_*.json` already contain the materialized episode lists (up to 2.4 MB for L4). A Python dataclass would duplicate `scenes`/`goals` (already in JSON) and could never carry the episode lists. JSON is source of truth; registry is a 4-line `dict[str, Path]` for short-name lookup. |
| 5 | `default_steps` location | **(5b) CLI default 2_400_000** | All 9 sbatch files pass `--steps` explicitly. The current default `10_000_000` is dead code. JSON migration adds nothing because sbatch always overrides. |
| 6 | Encoder abstraction | **(6α) `Encoder` ABC + inheritance** | CNN's encoder lives inside the agent (passthrough adapter); VGGT's encoder is an external `VGGTFeatureExtractor` that the adapter wraps. The asymmetry hides cleanly inside subclasses; `train()` just calls `encoder.make_adapter()`. Type-checker enforces every subclass implements the contract. |
| 7 | Test pyramid | **L1 + L2 + L3 + L4**, with **(A1) real GPU** + **(B1) hardcoded `PRESETS` list** | L1 structural / L2 construction / L3 adapter behavior / L4 preset-matrix-reaches-Trainer-init. GPU loading is fine because sessions run on H100. Hardcoded preset list (vs sbatch-parsing) avoids fragile shell parsing. |
| 8 | Test fixture source | **(8c) Recycle from `output/vggt/parity/`** | `input_frames_l1.npz` (10 frames @ 518×518×3) + `pt_outputs.npz` (validated VGGT outputs) already exist from #81's parity work. Copy to `modules/r2dreamer/launch/tests/fixtures/` and commit (~3.4 MB). |
| 9 | Per-level scripts vs `--curriculum` flag | **(9a) Per-level shim per `(encoder, level)`** | Each sbatch today couples `(encoder, level, output_dir, wandb_name)` manually. Putting the coupling in the shim's `train()` call prevents wrong-directory drift (production safety). Scaling to 28 shims with #82 is acceptable; each is 7 homogeneous lines. |
| 10 | Eval signature | **(10a) Symmetric to `train()`** — `evaluate(env, encoder, curriculum, ...)` | `modules/envs/habitat.py` already supports `curriculum_mode="train"\|"eval"`. Held-out-house eval (#76) becomes a new JSON in `data/curriculum/` + a registry entry, not a different code path. |
| 11 | Migration | **3 phases, 1 issue with checkboxes** | Phase 1 Foundation → Phase 2 Train+Eval (sharing plumbing, hence merged) → Phase 3 Parity+Archive. Each phase keeps green tests + working sbatch smokes. |
| 12 | Persistence | **(12c) Wiki page + Issue update** | This page is the design rationale; #85 carries the spec + checkboxes; #52 closes when Phase 3 lands. |

## Final folder structure

```
modules/r2dreamer/
  adapters/                              ← NEW (Phase 1)
    __init__.py
    obs_adapter.py                        # base ObsAdapter (moved from trainer.py)
    vggt_adapter.py                       # VGGTObsAdapter (moved from run_jax_habitat_vggt.py)
  launch/                                ← NEW (Phase 1+2+3)
    __init__.py
    encoders.py                           # Encoder ABC, CNNEncoder, VGGTEncoder        [P1]
    registries.py                         # encoder_registry, env_registry              [P1]
    curricula.py                          # CURRICULA: dict[str, Path] (4 entries)      [P1]
    habitat_setup.py                      # make_habitat_env factory                    [P1]
    parser.py                             # _build_parser_train(), _build_parser_eval() [P2]
    train.py                              # train(env, encoder, curriculum, ...)        [P2]
    evaluate.py                           # evaluate(env, encoder, curriculum, ...)     [P2]
    parity/                              ← NEW (Phase 3)
      __init__.py
      train_parity.py                     # parity_train(arch, jax_kwargs, pt_kwargs)
      benchmark.py                        # benchmark(arch, frameworks=[...])
      batch_utils.py                      # _convert_batch (JAX), make_batch_torch
    tests/                               ← NEW (Phase 1+2+3)
      __init__.py
      fixtures/
        sample_habitat_obs.npz            # recycled from output/vggt/parity/input_frames_l1.npz
        expected_vggt_outputs.npz         # recycled from output/vggt/parity/pt_outputs.npz
      test_registries.py                  # L1 structural                               [P1]
      test_encoders.py                    # L2 construction + L3 adapter behavior       [P1]
      test_presets.py                     # L4 preset-matrix-reaches-Trainer-init       [P2]
  scripts/                                # 8 per-level shims (7 lines each)            [P2]
    run_jax_habitat_l1.py
    run_jax_habitat_l2.py
    run_jax_habitat_l3.py
    run_jax_habitat_l4.py
    run_jax_habitat_vggt_l1.py
    run_jax_habitat_vggt_l2.py            # (added when L2-vggt sbatch lands)
    run_jax_crafter.py                    # 3 lines, no level
    eval_habitat.py                       # 7 lines shim into evaluate()                [P2]

archiv/r2dreamer-pytorch-20260425/       ← NEW (Phase 3)
  run_pytorch_crafter.py                  # 160 lines — Hydra subprocess wrapper
  run_pytorch_standalone.py               # 354 lines — standalone PT training loop
  run_pytorch_comparison.sh               # invokes the above
  run_pytorch_100k.sh                     # invokes the above
```

## Encoder ABC sketch

```python
# modules/r2dreamer/launch/encoders.py
from abc import ABC, abstractmethod
from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from modules.vggt.feature_extractor import VGGTFeatureExtractor


class Encoder(ABC):
    """Base class for everything an agent might consume as input."""

    @abstractmethod
    def make_adapter(self) -> ObsAdapter:
        """Return the ObsAdapter that bridges env obs to agent input."""


class CNNEncoder(Encoder):
    """Identity encoder — agent's internal CNN handles RGB → embedding → RSSM."""

    def make_adapter(self) -> ObsAdapter:
        return ObsAdapter()  # passthrough, default behavior


class VGGTEncoder(Encoder):
    """External feature extractor — 518×518 RGB → 4116-dim flat vector."""

    def __init__(self, resolution: int = 518):
        self._extractor = VGGTFeatureExtractor()  # device="cuda" default

    def make_adapter(self) -> ObsAdapter:
        return VGGTObsAdapter(self._extractor)


encoder_registry: dict[str, type[Encoder]] = {
    "cnn":  CNNEncoder,
    "vggt": VGGTEncoder,
    # Future #82 variants: "vggt_224", "vggt_distill", "vggt_semdpt", "vggt_async"
}
```

## Per-level shim convention

```python
# modules/r2dreamer/scripts/run_jax_habitat_vggt_l2.py — 11 lines
"""L2 VGGT shim — habitat, vggt, L2."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="vggt", curriculum="L2",
        output_dir="output/r2dreamer-curriculum-l2-vggt",
        wandb_name="r2d-L2-vggt",
        wandb_tags=["curriculum", "level2", "vggt"],
    )
```

The `sys.path.insert(...)` boilerplate is required because invoking `python script.py` directly only puts `script.py`'s directory on `sys.path`, not the repo root — `from modules.r2dreamer.launch.train import train` would fail otherwise. Pytest hides this issue (it configures `sys.path` via `[tool.pytest.ini_options]`), so unit tests pass even when shims are broken; only end-to-end smoke (`python script.py --steps 1000`) catches it.

CLI flags `--steps`, `--prefill`, `--output_dir`, `--wandb_name`, `--wandb_tags` remain available as overrides. Sbatch files become minimal:

```bash
#SBATCH --output=output/r2dreamer-curriculum-l2-vggt/slurm-%j.out
mkdir -p output/r2dreamer-curriculum-l2-vggt

uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt_l2.py \
    --steps 2400000 --prefill 5000
```

## Test strategy

| Layer | What it checks | Cost |
|---|---|---|
| **L1 — structural** (`test_registries.py`) | Every `encoder_registry` value is `type[Encoder]`; every Encoder subclass implements `make_adapter`; KeyError lists valid names | <1 sec |
| **L2 — construction** (`test_encoders.py`) | `CNNEncoder()` constructs; `VGGTEncoder()` constructs (loads model on GPU); `make_adapter()` returns an `ObsAdapter` | CNN: <1 sec; VGGT: ~10-30 sec, marked `@pytest.mark.gpu` |
| **L3 — adapter behavior** (`test_encoders.py`) | Load `sample_habitat_obs.npz` (10 real Habitat frames). For VGGT: assert adapter output matches `expected_vggt_outputs.npz` (flattened) within bf16 tolerance. For CNN: assert passthrough shape/dtype. | GPU-bound for VGGT |
| **L4 — preset matrix** (`test_presets.py`) | Hardcoded `PRESETS = [(env, encoder, curriculum), ...]` covering every active sbatch combo. Each instantiates `train(...)` up to (but not including) `trainer.fit()` — i.e. tests that wiring reaches `Trainer.__init__` without crash. | ~30 sec/preset, run on demand |

The fixture comes from `output/vggt/parity/`:
- `input_frames_l1.npz` — 10 frames `(10, 3, 518, 518) uint8`
- `pt_outputs.npz` — `world_points (10, 37, 37, 3) float32` + `camera_pose (10, 9) float32`, flattened per-frame to `(4116,)` for adapter comparison

GPU-marked tests register the `gpu` marker in `pyproject.toml` so `pytest -m "not gpu"` skips them on CPU-only hosts.

## Migration plan — 3 phases, 1 issue, 1 PR per phase

### Phase 1 — Foundation

**Scope:**
- Create `modules/r2dreamer/adapters/` package with `obs_adapter.py` (moved from `trainer.py`) + `vggt_adapter.py` (moved from `run_jax_habitat_vggt.py`).
- Create `modules/r2dreamer/launch/` skeleton: `encoders.py`, `registries.py`, `curricula.py`, `habitat_setup.py`.
- Copy `output/vggt/parity/{input_frames_l1,pt_outputs}.npz` to `tests/fixtures/`.
- Create L1 + L2 + L3 tests in `tests/`.

**Non-changes:** Existing scripts (`run_jax_habitat.py`, etc.) stay untouched. They keep working via direct imports. No sbatch edits.

**Exit gate:** All new tests green. Existing `pytest modules/r2dreamer/tests/` still green (the move of `ObsAdapter` and `VGGTObsAdapter` requires updating the existing imports — `trainer.py`, `run_jax_habitat_vggt.py` — but their behavior is unchanged).

### Phase 2 — Train + Eval

**Scope:**
- Create `launch/parser.py` (`_build_parser_train`, `_build_parser_eval`).
- Create `launch/train.py` and `launch/evaluate.py`.
- Rewrite the 3 existing r2dreamer scripts as 7-line shims; add 5+ new per-level shims (`run_jax_habitat_l{1,2,3,4}.py`, `run_jax_habitat_vggt_l1.py`, etc., one per active sbatch combo).
- Rewrite `eval_habitat.py` as a shim.
- Edit 9 sbatch files: replace `--curriculum_path data/curriculum/level{N}_*.json` with `--curriculum L{N}`, change script paths to per-level shims.
- Add L4 preset-matrix tests.

**Non-changes:** Pytorch scripts still in `modules/r2dreamer/scripts/` (not yet archived). Parity scripts (`run_parity_training.py`, `run_benchmark.py`) untouched.

**Exit gate:** All new tests green. **Smoke each of the 9 active sbatch jobs with `--steps 1000`**; each must reach the training loop and write 1 checkpoint without crash. Bench against the L1 baseline metrics from #81 (within ±5%).

### Phase 3 — Parity + Archive

**Scope:**
- Create `launch/parity/` sub-package: `train_parity.py`, `benchmark.py`, `batch_utils.py`.
- Rewrite `run_parity_training.py`, `run_benchmark.py` as shims.
- `git mv` PyTorch scripts: `run_pytorch_crafter.py`, `run_pytorch_standalone.py`, `run_pytorch_comparison.sh`, `run_pytorch_100k.sh` → `archiv/r2dreamer-pytorch-20260425/`.
- Close #52 with a comment listing the absorption: "Training entrypoint duplication absorbed by #85 Phase 2; PyTorch script duplication archived in Phase 3; remaining `eval_habitat.py` boilerplate absorbed in Phase 2 (shim into `evaluate()`)."
- Close #85.

**Exit gate:** All tests green. `gh pr` cross-checks: no dangling references to archived scripts in active sbatch / wiki / tests.

## Out of scope

- **No Hydra, OmegaConf, or YAML config files.** (RFC's original "What This RFC Does NOT Do" stands.)
- **No registries for loggers, optimizers, or replay buffers.** Single implementation each; speculative.
- **No new framework axis on `train()`.** PyTorch scripts go to archive; framework-axis would only exist for variants we don't keep.
- **No `EncoderSpec` dataclass or `agent_input_kind` field.** The Encoder ABC is the abstraction; subclasses encode their own kind via what they put in their adapter.
- **No combinatorial preset matrix from `encoder_registry × env_registry × CURRICULA`.** Only test combinations that real sbatch files use (avoids testing nonsense like `crafter + L4`).

## Related

- [#85](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/85) — original RFC; this page is the agreed final design.
- [#52](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/52) — extract shared boilerplate; subsumed.
- [#82](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/82) — alternate VGGT speedup strategies; new encoder variants from #82 plug in as new `Encoder` subclasses + registry entries.
- [#76](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/76) — held-out-house eval; becomes a new JSON in `data/curriculum/` + a `CURRICULA` entry.
- [#84](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/84) — `HabitatObjectNavEnv` deepening; `make_habitat_env` factory will benefit when #84 lands.
- [methods/training-orchestration.md](training-orchestration.md) — the Trainer module + ObsAdapter pattern this builds on.
