"""Tests for Habitat environment. Skipped if habitat-sim is not installed."""

import pytest

try:
    import habitat_sim

    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False

skip_no_habitat = pytest.mark.skipif(
    not HAS_HABITAT,
    reason="habitat-sim not installed — run: uv sync --extra habitat",
)


@skip_no_habitat
def test_habitat_sim_import():
    import habitat_sim  # noqa: F811

    assert hasattr(habitat_sim, "SimulatorConfiguration")


@skip_no_habitat
def test_numpy_version():
    import numpy as np
    from packaging.version import Version

    assert Version(np.__version__) >= Version("2.0"), (
        f"numpy override failed: got {np.__version__}, expected >=2.0"
    )


@skip_no_habitat
def test_habitat_sim_configuration():
    import habitat_sim  # noqa: F811

    cfg = habitat_sim.SimulatorConfiguration()
    assert cfg is not None

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [64, 64]
    assert rgb_sensor.uuid == "rgb"

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        sensor_specifications=[rgb_sensor]
    )
    assert len(agent_cfg.sensor_specifications) == 1
