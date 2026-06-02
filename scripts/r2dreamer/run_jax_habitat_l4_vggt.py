"""L4 VGGT shim — habitat, vggt, L4 (10 houses, 6 goals, full curriculum, 3D encoder).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l4-vggt"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l4-vggt")
