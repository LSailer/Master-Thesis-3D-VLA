"""Shared R2Dreamer encoder type names."""

FLAT_VGGT_ENCODER_TYPES = (
    "vggt",
    "vggt_aggregator_mlp",
    "vggt_agg_token_transformer",
    "vggt_wp_dense_cnn",
    "vggt_wp_cp_64",
)

EVAL_ENCODER_TYPES = (
    "cnn",
    *FLAT_VGGT_ENCODER_TYPES,
    "hybrid",
    "vggt_house_context",
    "vggt_house_points_pose",
    "vggt_house_full_tokens_nogate",
    "vggt_house_global_tokens_nogate",
)
