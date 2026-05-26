# Archived legacy sbatch scripts (3D-34 / s5)

These hand-written sbatch scripts were superseded by the YAML-backed Slurm
launcher (`scripts/slurm/`, parent issue 3D-29). They are kept here, not
deleted, because the curriculum scripts remain the **frozen golden references**
for the launcher's render-equivalence tests (`tests/slurm/test_launch.py`).

To launch any of these jobs today, use the launcher instead:

```bash
bash scripts/slurm/launch.sh <variant> --smoke-then-prod
```

| Archived file                          | Original location              | Replaced by (`launch.sh <variant>`) | Migrated in |
|----------------------------------------|--------------------------------|-------------------------------------|-------------|
| `train_curriculum_l1_vggt.sbatch`      | `scripts/r2dreamer/slurm/`     | `l1_vggt`                           | 3D-30 (s1)  |
| `train_curriculum_l2_vggt.sbatch`      | `scripts/r2dreamer/slurm/`     | `l2_vggt`                           | 3D-31 (s2)  |
| `train_curriculum_l3_vggt.sbatch`      | `scripts/r2dreamer/slurm/`     | `l3_vggt`                           | 3D-31 (s2)  |
| `train_curriculum_l4_vggt.sbatch`      | `scripts/r2dreamer/slurm/`     | `l4_vggt`                           | 3D-31 (s2)  |
| `prod_aggregator_mlp_v1.sbatch`        | `scripts/`                     | `aggregator_mlp_v1` (prod)          | 3D-32 (s3)  |
| `smoke_aggregator_mlp_fast_path.sbatch`| `scripts/`                     | `aggregator_mlp_v1` (smoke)         | 3D-32 (s3)  |
| `collect_offline_buffer_3d25.sbatch`   | `scripts/r2dreamer/slurm/`     | `offline_buffer_3d25`               | 3D-33 (s4)  |
| `submit_offline_buffer.sh`             | `scripts/r2dreamer/slurm/`     | `launch.sh offline_buffer_3d25 --smoke-then-prod` | 3D-33 (s4) |

`submit_offline_buffer.sh` still resolves `collect_offline_buffer_3d25.sbatch`
relative to its own directory, so it remains runnable from this folder if needed.

**Not archived:** the non-migrated legacy scripts (`*_actfix`, `*_rerun`,
`train_curriculum_l{1,2,3,4}.sbatch` (non-vggt), `*_resume*`,
`train_curriculum_l1_vggt_aggregator_mlp_2m.sbatch`, `smoke_curriculum_vggt.sbatch`,
`profile_pipeline_aggregator_mlp.sbatch`) stay in their original locations — they
have no YAML replacement yet and need separate evaluation (see 3D-34).
