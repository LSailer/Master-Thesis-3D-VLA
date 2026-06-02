"""L2 CNN shim — habitat, cnn, L2 (1 house, 6 goals).

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l2-cnn"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l2-cnn")
