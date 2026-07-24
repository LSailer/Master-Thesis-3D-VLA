"""Parametrized CPU test over the full encoder RECIPES registry.

For every registered recipe: build the recipe's init dummy (never load VGGT),
infer the obs spec from it, run the composite fail-fast key check where it
applies, init the module, and assert ``apply`` preserves a replay ``(B, T)``
prefix with finite outputs. This is the generic shape/plumbing contract of
HANDOFF verification; ``cnn``/``hybrid`` numeric parity lives in
``test_composite_encoder.py`` and the golden run.
"""

import jax
import jax.numpy as jnp
import pytest

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.composite import CompositeEncoder
from src.r2dreamer.encoders.constants import HYBRID_RGB_DIM
from src.r2dreamer.encoders.gnn_house import GnnHousePointsCameraEncoder
from src.r2dreamer.encoders.recipes import (
    RECIPES,
    build_encoder_module,
    check_branch_keys,
    dummy_encoder_obs,
    infer_obs_spec,
)
from src.r2dreamer.observation_keys import (
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
)

# Per-type minimal configs sized for CPU init/apply. Kept next to the test so a
# new recipe must add a row here (the registry-coverage test enforces it).
SMALL_TOKEN_KW = dict(
    vggt_token_count=10,
    vggt_token_dim=16,
    vggt_token_projection_dim=32,
    vggt_token_transformer_heads=4,
    vggt_token_transformer_layers=1,
    vggt_embed_dim=64,
)
CONFIGS: dict[str, dict] = {
    "cnn": dict(obs_shape=(64, 64, 3)),
    "hybrid": dict(
        obs_shape=(HYBRID_RGB_DIM + 4116,),
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1),
        vggt_embed_dim=16,
        mlp_vggt_hidden=16,
        mlp_vggt_layers=1,
    ),
    "vggt": dict(obs_shape=(4116,)),
    "vggt_wp_cp_64": dict(obs_shape=(12297,)),
    "vggt_aggregator_mlp": dict(obs_shape=(3072,)),
    "vggt_agg_raw": dict(obs_shape=(1370 * 1024,)),
    "vggt_agg_token_transformer": dict(obs_shape=(160,), **SMALL_TOKEN_KW),
    "vggt_wp_dense_cnn": dict(obs_shape=(70, 70, 3), vggt_embed_dim=64),
    "vggt_wp64_cnn_cp_mlp": dict(
        obs_shape={"world_points": (64, 64, 3), "camera_pose": (9,)},
        encoder_depth=4,
        encoder_mults=(2, 2, 2, 2),
        vggt_embed_dim=64,
        mlp_vggt_hidden=32,
        mlp_vggt_layers=1,
    ),
    "vggt_house_context": dict(
        obs_shape=(HYBRID_RGB_DIM + 1024,),
        vggt_feature_dim=1024,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1),
        vggt_embed_dim=16,
        mlp_vggt_hidden=16,
        mlp_vggt_layers=1,
    ),
    "vggt_house_points_pose": dict(
        obs_shape={HOUSE_CONTEXT_KEY: (64, 6)},
    ),
    "vggt_hybrid_house_points_pose": dict(
        obs_shape={HOUSE_CONTEXT_KEY: (64, 6)},
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1),
    ),
    "gnn_house_points_pose": dict(obs_shape={HOUSE_CONTEXT_KEY: (64, 6)}),
    "gnn_edge_house_points_pose": dict(obs_shape={HOUSE_CONTEXT_KEY: (64, 6)}),
    "vggt_house_full_tokens_nogate": dict(obs_shape=(64, 64, 3), **SMALL_TOKEN_KW),
    "vggt_house_global_tokens_nogate": dict(obs_shape=(64, 64, 3), **SMALL_TOKEN_KW),
    "vggt_house_global_embedding": dict(
        obs_shape={HYBRID_IMAGE_KEY: (64, 64, 3), GLOBAL_PATCH_TOKENS_KEY: (8, 16)},
    ),
}

# Recipes whose init is skipped on CPU (construction is still asserted).
INIT_SKIP = {
    # First Dense kernel is (1370*1024, 1024) float32 ≈ 5.7 GB.
    "vggt_agg_raw": "AGG_RAW first-layer kernel too large for CPU init",
    # GNN k-NN graph construction is exercised by test_gnn_house_encoder.py.
    "gnn_house_points_pose": "GNN graph init covered by its dedicated tests",
    "gnn_edge_house_points_pose": "GNN graph init covered by its dedicated tests",
}


def _config_for(name: str) -> R2DreamerConfig:
    kwargs = dict(CONFIGS[name])
    if name.startswith("gnn_"):
        kwargs["encoder_module_cls"] = GnnHousePointsCameraEncoder
    return R2DreamerConfig(encoder_type=name, num_actions=4, **kwargs)


def test_registry_covers_every_config_row_and_vice_versa():
    assert set(RECIPES) == set(CONFIGS)


def test_gnn_requires_module_cls_override():
    cfg = R2DreamerConfig(
        encoder_type="gnn_house_points_pose",
        obs_shape={HOUSE_CONTEXT_KEY: (64, 6)},
        num_actions=4,
    )
    with pytest.raises(ValueError, match="unknown encoder_type"):
        build_encoder_module(cfg)


def test_unknown_encoder_type_raises():
    cfg = R2DreamerConfig(encoder_type="not_a_real_encoder")
    with pytest.raises(ValueError, match="unknown encoder_type"):
        build_encoder_module(cfg)


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe_dummy_infer_init_apply(name):
    recipe = RECIPES[name]
    cfg = _config_for(name)

    module = recipe.build_module(cfg)
    assert module is not None

    frame = dummy_encoder_obs(cfg)
    obs_spec = infer_obs_spec(frame)
    if isinstance(module, CompositeEncoder):
        # Decision 5: the single startup check, for composite encoders.
        check_branch_keys(module.spec, obs_spec.keys())

    if name in INIT_SKIP:
        pytest.skip(INIT_SKIP[name])

    params = module.init(jax.random.PRNGKey(0), frame)

    # Replay-shaped batch: (B, T, *event) per key; singleton/scalar context
    # fields (house cloud snapshot + its size) keep their batch-1 layout.
    B, T = 2, 3

    def replay_like(key, value):
        # Singleton context fields (house cloud + size, global-token context)
        # keep their batch-1 layout — they are joined, not replay-batched.
        if key in (HOUSE_CONTEXT_KEY, HOUSE_CONTEXT_SIZE_KEY, GLOBAL_TOKENS_KEY):
            return value
        return jnp.zeros((B, T, *value.shape[1:]), value.dtype)

    if isinstance(frame, dict):
        batch = {k: replay_like(k, v) for k, v in frame.items()}
    else:
        batch = jnp.zeros((B, T, *frame.shape[1:]), frame.dtype)

    out = module.apply(params, batch)
    assert out.shape[:2] == (B, T)
    assert out.shape[-1] > 0
    assert bool(jnp.all(jnp.isfinite(out)))


def test_rgb_key_marks_decoder_capable_recipes():
    rgb_capable = {name for name, r in RECIPES.items() if r.rgb_key is not None}
    assert rgb_capable == {
        "cnn",
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
        "vggt_house_global_embedding",
    }


def test_check_branch_keys_rejects_mismatch():
    composite = RECIPES["hybrid"].composite(_config_for("hybrid"))
    with pytest.raises(ValueError, match="key mismatch"):
        check_branch_keys(composite, {HYBRID_IMAGE_KEY})  # missing wp_cp
