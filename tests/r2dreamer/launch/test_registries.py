"""L1 structural tests: Observation Preparation registry and curriculum paths."""

import inspect
import pytest

from src.r2dreamer.encoders import Encoder, HybridEncoder
from src.r2dreamer.launch.registries import (
    env_registry,
    observation_preparation_registry,
)
from src.r2dreamer.launch.curricula import CURRICULA


class TestObservationPreparationRegistry:
    def test_all_values_are_encoder_subclasses(self):
        for name, cls in observation_preparation_registry.items():
            assert inspect.isclass(cls), f"{name!r} is not a class"
            assert issubclass(cls, Encoder), f"{name!r} is not a subclass of Encoder"

    def test_all_encoder_subclasses_implement_make_adapter(self):
        for name, cls in observation_preparation_registry.items():
            assert hasattr(cls, "make_adapter"), f"{name!r} missing make_adapter"
            assert callable(cls.make_adapter), f"{name!r}.make_adapter not callable"

    def test_missing_key_raises_key_error(self):
        with pytest.raises(KeyError):
            _ = observation_preparation_registry["nonexistent"]

    def test_known_keys_present(self):
        assert "cnn" in observation_preparation_registry
        assert "vggt" in observation_preparation_registry
        assert "vggt_aggregator_mlp" in observation_preparation_registry
        assert "vggt_wp_dense_cnn" in observation_preparation_registry
        assert "hybrid" in observation_preparation_registry

    def test_hybrid_key_resolves_to_hybrid_spec_class(self):
        assert observation_preparation_registry["hybrid"] is HybridEncoder


class TestEnvRegistry:
    def test_all_values_are_callable(self):
        for name, factory in env_registry.items():
            assert callable(factory), f"{name!r} factory is not callable"

    def test_known_keys_present(self):
        assert "habitat" in env_registry
        assert "crafter" in env_registry


class TestCurricula:
    def test_all_curriculum_paths_exist(self):
        for name, path in CURRICULA.items():
            assert path.exists(), f"Curriculum {name!r} path does not exist: {path}"

    def test_expected_keys(self):
        assert set(CURRICULA.keys()) == {"L1", "L2", "L3", "L4"}
