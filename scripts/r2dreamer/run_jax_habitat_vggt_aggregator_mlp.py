"""L1 Variant 1 shim — habitat + VGGT aggregator MLP encoder.

Run metadata lives in _run_configs.RUN_CONFIGS["habitat-l1-vggt-aggregator-mlp"].
"""
import _run_configs

if __name__ == "__main__":
    _run_configs.launch_run("habitat-l1-vggt-aggregator-mlp")
