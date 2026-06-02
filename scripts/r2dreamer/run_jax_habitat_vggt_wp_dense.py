"""L1 VGGT shim — habitat, full-resolution world-point CNN encoder (3D-53).

Feeds the dense 518x518x3 VGGT world-point map (no 37x37 pooling) into a conv
encoder that treats XYZ as a 3-channel image. Counterpart to the WP/CP MLP run.

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l1-vggt-wp-dense"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l1-vggt-wp-dense")
