"""L1 CNN shim — habitat, cnn, L1 (1 house, chair only).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l1-cnn"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l1-cnn")
