"""L4 CNN shim — habitat, cnn, L4 (10 houses, 6 goals, full curriculum).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l4-cnn"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l4-cnn")
