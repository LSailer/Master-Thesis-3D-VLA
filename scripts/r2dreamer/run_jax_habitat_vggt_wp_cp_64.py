"""L1 VGGT shim — habitat, WP+CP MLP at a 64x64 world-point grid (3D-52/3D-53).

Same MLP / camera-pose / replay setup as run_jax_habitat_vggt.py, but VGGT's
dense point map is pooled to 64x64 (obs = 64*64*3 + 9 = 12297) instead of 37x37.
Controlled resolution ablation vs the 37x37 WP+CP MLP run.

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l1-vggt-wp-cp-64"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l1-vggt-wp-cp-64")
