"""Crafter shim — crafter, cnn, no curriculum.

Run metadata lives in _run_configs.RUN_CONFIGS["crafter-cnn"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("crafter-cnn")
