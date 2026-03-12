"""AI2-THOR controller init smoke test.

Requires headless display (Xvfb) and GPU access.
Run via: xvfb-run -a uv run pytest tests/test_ai2thor_smoke.py -x -q
"""

import os
import pytest


@pytest.mark.gpu
class TestAI2THORController:
    def test_controller_init_and_step(self):
        """Controller init with CUDA_VISIBLE_DEVICES workaround, Pass action succeeds."""
        # BWUniCluster quirk: vulkaninfo missing → must unset before Controller init
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        from ai2thor.controller import Controller

        # Use DISPLAY set by xvfb-run (e.g. ":99"), strip leading ":"
        display = os.environ.get("DISPLAY", ":0")
        x_display = display.lstrip(":")

        # headless=False: let Unity render to the Xvfb virtual display
        controller = Controller(
            scene="FloorPlan1",
            width=300,
            height=300,
            x_display=x_display,
        )
        try:
            event = controller.step("Pass")
            assert event.metadata["lastActionSuccess"], (
                f"Pass action failed: {event.metadata.get('errorMessage', '')}"
            )
        finally:
            controller.stop()
