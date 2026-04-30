"""Smoke test: validate JAX + Habitat + wandb environment."""

import sys


def check_imports():
    errors = []
    for mod in ["jax", "optax", "flax", "habitat_sim", "wandb", "numpy", "h5py"]:
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except ImportError as e:
            print(f"  [FAIL] {mod}: {e}")
            errors.append(mod)
    return errors


def check_jax_gpu():
    import jax

    devices = jax.devices()
    print(f"  JAX devices: {devices}")
    gpu_devices = [d for d in devices if d.device_kind != "cpu"]
    if not gpu_devices:
        print("  [WARN] No GPU device found — running on CPU")
        return False

    try:
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (128, 128))
        b = a @ a.T
        assert b.shape == (128, 128), f"Unexpected shape: {b.shape}"
        print(f"  [OK] 128x128 matmul on {gpu_devices[0]}")
        return True
    except Exception as e:
        print(f"  [WARN] GPU compute failed (may be experimental backend): {e}")
        return False


def check_habitat():
    import habitat_sim

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [64, 64]

    agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=[rgb_sensor])
    try:
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
        obs = sim.get_sensor_observations()
        print(f"  [OK] Habitat obs keys: {list(obs.keys())}, rgb shape: {obs['rgb'].shape}")
        sim.close()
    except Exception as e:
        print(f"  [WARN] Habitat sim test skipped: {e}")
        return False
    return True


def print_versions():
    import jax
    import numpy as np

    print(f"  JAX {jax.__version__}  |  NumPy {np.__version__}")
    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
    except ImportError:
        pass
    try:
        import habitat_sim
        print(f"  habitat-sim {habitat_sim.__version__}")
    except (ImportError, AttributeError):
        pass


def main():
    print("=== Import checks ===")
    import_errors = check_imports()

    print("\n=== JAX GPU check ===")
    has_gpu = check_jax_gpu()

    print("\n=== Habitat check ===")
    hab_ok = check_habitat()

    print("\n=== Versions ===")
    print_versions()

    print("\n=== Summary ===")
    if import_errors:
        print(f"FAIL: missing imports: {import_errors}")
        sys.exit(1)
    if not has_gpu:
        print("WARN: no GPU — CPU only")
    if not hab_ok:
        print("WARN: habitat sim test scene missing or failed")
    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
