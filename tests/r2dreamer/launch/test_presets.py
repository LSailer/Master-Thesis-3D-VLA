
"""L4 preset-matrix tests — verify every active sbatch combo resolves through registries."""

import pytest

# Every (env, encoder, curriculum) combination in active sbatch files.
PRESETS = [
    ("habitat", "cnn",  "L1"),
    ("habitat", "cnn",  "L2"),
    ("habitat", "cnn",  "L3"),
    ("habitat", "cnn",  "L4"),
    ("habitat", "vggt", "L1"),
    ("habitat", "hybrid", "L1"),
    ("habitat", "vggt_house_points_pose", "L1"),
    ("habitat", "vggt_house_full_tokens_nogate", "L1"),
    ("habitat", "vggt_house_global_tokens_nogate", "L1"),
    ("habitat", "vggt_house_global_embedding", "L1"),
    ("crafter", "cnn",  None),
]


@pytest.mark.parametrize("env_name, encoder_name, curriculum_name", PRESETS)
def test_preset_resolves(env_name, encoder_name, curriculum_name):
    """Every (env, encoder, curriculum) combo from active sbatch files must
    resolve through registries + curricula without raising."""
    from src.environments.habitat import HABITAT_CURRICULA
    from src.r2dreamer.launch.registries import env_registry, encoder_registry

    assert env_name in env_registry, f"{env_name!r} not in env_registry"
    assert encoder_name in encoder_registry, f"{encoder_name!r} not in encoder_registry"
    if curriculum_name is not None:
        assert curriculum_name in HABITAT_CURRICULA, (
            f"{curriculum_name!r} not in HABITAT_CURRICULA"
        )
        curriculum_path = HABITAT_CURRICULA[curriculum_name]
        if not curriculum_path.parent.exists():
            pytest.skip(
                f"curriculum data not provisioned ({curriculum_path.parent} absent); "
                "generate it with scripts/environments/generate_curriculum.py"
            )
        assert curriculum_path.exists(), f"Curriculum file missing: {curriculum_path}"


# Full end-to-end wiring test is deferred: constructing HabitatObjectNavEnv requires
# Habitat-Sim scene files and GPU, making it unsuitable for a unit test.
# The resolve-only test above is sufficient for CI; sbatch smoke tests (--steps 1000)
# serve as the integration gate per Phase 2 exit criteria.
