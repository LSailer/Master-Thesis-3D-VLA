"""L1 Hybrid shim — habitat + CNN(RGB) + gated MLP(WP/CP) hybrid encoder (3D-50/51/52).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l1-hybrid"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l1-hybrid")
