"""L1 VGGT shim — habitat, vggt, L1 (1 house, chair only, 3D encoder).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l1-vggt"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l1-vggt")
