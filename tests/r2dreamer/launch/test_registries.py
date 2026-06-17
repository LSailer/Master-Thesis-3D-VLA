"""L1 structural tests — registry contents and curriculum paths."""

import inspect
import pytest

from src.r2dreamer.encoders import Encoder, HybridEncoder
from src.r2dreamer.launch.registries import encoder_registry, env_registry
from src.r2dreamer.launch.curricula import CURRICULA


class TestEncoderRegistry:
    def test_all_values_are_encoder_subclasses(self):
        for name, cls in encoder_registry.items():
            assert inspect.isclass(cls), f"{name!r} is not a class"
            assert issubclass(cls, Encoder), f"{name!r} is not a subclass of Encoder"

    def test_all_encoder_subclasses_implement_make_adapter(self):
        for name, cls in encoder_registry.items():
            assert hasattr(cls, "make_adapter"), f"{name!r} missing make_adapter"
            assert callable(cls.make_adapter), f"{name!r}.make_adapter not callable"

    def test_missing_key_raises_key_error(self):
        with pytest.raises(KeyError):
            _ = encoder_registry["nonexistent"]

    def test_known_keys_present(self):
        assert "cnn" in encoder_registry
        assert "vggt" in encoder_registry
        assert "vggt_aggregator_mlp" in encoder_registry
        assert "vggt_agg_token_transformer" in encoder_registry
        assert "vggt_wp_dense_cnn" in encoder_registry
        assert "vggt_wp64_cnn_cp_mlp" in encoder_registry
        assert "hybrid" in encoder_registry
        assert "vggt_house_context" in encoder_registry
        assert "vggt_house_full_tokens_nogate" in encoder_registry

    def test_hybrid_key_resolves_to_hybrid_spec_class(self):
        assert encoder_registry["hybrid"] is HybridEncoder


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
