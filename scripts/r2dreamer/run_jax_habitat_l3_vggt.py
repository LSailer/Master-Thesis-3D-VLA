"""L3 VGGT shim — habitat, vggt, L3 (10 houses, chair only, 3D encoder).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l3-vggt"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l3-vggt")
