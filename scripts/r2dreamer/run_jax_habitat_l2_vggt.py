"""L2 VGGT shim — habitat, vggt, L2 (1 house, 6 goals, 3D encoder).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l2-vggt"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l2-vggt")
