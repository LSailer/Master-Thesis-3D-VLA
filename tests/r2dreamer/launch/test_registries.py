"""L1 structural tests — registry contents and curriculum paths."""

import inspect
import pytest

from src.configs.agent_config import R2DreamerConfig
from src.environments.habitat import (
    HABITAT_CURRICULA,
    HabitatEnvConfig,
    resolve_habitat_curriculum_path,
)
from src.r2dreamer.encoders import Encoder, HybridEncoder
from src.r2dreamer.encoders.factory import _make_encoder
from src.r2dreamer.launch.registries import encoder_registry, env_registry
from src.r2dreamer.observation_preparation.contracts import (
    encoder_module_kwargs_from_config,
)


def _registry_module_cls(encoder_cls: type[Encoder]) -> type:
    """Resolve an Encoder selection's module class without running __init__.

    ``module_cls`` is either a plain class attribute or a property that only
    reads class-level state (the ``variant`` descriptor), so an uninitialized
    instance suffices — full construction would build the VGGT extractor.

    Args:
      encoder_cls: A launcher-side ``Encoder`` selection class.

    Returns:
      The Flax Encoder Module class the selection would instantiate.
    """
    attr = inspect.getattr_static(encoder_cls, "module_cls")
    if isinstance(attr, property):
        return attr.fget(object.__new__(encoder_cls))
    return attr


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
        assert "vggt_house_points_pose" in encoder_registry
        assert "vggt_hybrid_house_points_pose" in encoder_registry
        assert "vggt_house_full_tokens_nogate" in encoder_registry
        assert "vggt_house_global_tokens_nogate" in encoder_registry

    def test_hybrid_key_resolves_to_hybrid_spec_class(self):
        assert encoder_registry["hybrid"] is HybridEncoder

    @pytest.mark.parametrize(
        "encoder_type",
        sorted(encoder_registry),
    )
    def test_module_constructs_from_contract_kwargs(self, encoder_type):
        # Regression: subclass modules (e.g. the GNN variants of
        # HousePointsCameraEncoder) must dispatch to their parent's kwargs
        # branch, not the generic MLP tail, on contract-snapshot paths
        # (make_encoder_module, checkpoint eval) that bypass agent.py.
        module_cls = _registry_module_cls(encoder_registry[encoder_type])
        config = R2DreamerConfig(encoder_type=encoder_type)

        kwargs = encoder_module_kwargs_from_config(config, module_cls)
        module = module_cls(**kwargs)

        assert isinstance(module, module_cls)

    @pytest.mark.parametrize(
        "encoder_type",
        [
            # A variant-driven (VGGT dispatch) and a standalone launcher
            # encoder. Neither applies a compute_dtype overlay when
            # ``full_bf16`` is off, so the factory fresh-build kwargs must
            # equal the contract-snapshot resolver kwargs exactly.
            "vggt",
            "vggt_house_global_embedding",
        ],
    )
    def test_factory_and_resolver_agree_on_kwargs(self, encoder_type):
        # Regression for the encoder-kwargs consolidation: the factory
        # ``_make_*`` builders delegate to the launcher
        # ``module_kwargs_from_config`` (the same path the contract-snapshot
        # resolver takes), so the two cannot diverge on no-dtype encoders.
        config = R2DreamerConfig(encoder_type=encoder_type)
        assert config.full_bf16 is False  # no compute_dtype overlay below

        module = _make_encoder(config)
        resolver_kwargs = encoder_module_kwargs_from_config(config)

        for key, value in resolver_kwargs.items():
            assert getattr(module, key) == value, (
                f"{encoder_type}: module.{key}={getattr(module, key)!r} "
                f"!= resolver {value!r}"
            )


class TestEnvRegistry:
    def test_all_values_are_callable(self):
        for name, factory in env_registry.items():
            assert callable(factory), f"{name!r} factory is not callable"

    def test_known_keys_present(self):
        assert "habitat" in env_registry
        assert "crafter" in env_registry


class TestHabitatCurricula:
    def test_all_curriculum_paths_exist(self):
        for name, path in HABITAT_CURRICULA.items():
            assert path.exists(), f"Curriculum {name!r} path does not exist: {path}"

    def test_expected_keys(self):
        assert set(HABITAT_CURRICULA.keys()) == {"L1", "L2", "L3", "L4"}

    def test_config_resolves_named_curriculum(self):
        config = HabitatEnvConfig(curriculum="L1")

        assert resolve_habitat_curriculum_path(config) == HABITAT_CURRICULA["L1"]

    def test_config_curriculum_path_overrides_name(self, tmp_path):
        override = tmp_path / "curriculum.json"
        config = HabitatEnvConfig(curriculum="L1", curriculum_path=override)

        assert resolve_habitat_curriculum_path(config) == override
