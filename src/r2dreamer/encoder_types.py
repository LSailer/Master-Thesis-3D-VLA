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
    "vggt_house_global_embedding",
)

# Encoders whose observation packs RGB alongside other modalities, so the RGB
# has to be extracted (by key when the obs is a Mapping, else by slicing the
# leading 3*64*64 channels) before it can serve as a decoder target.
COMPOSITE_RGB_ENCODER_TYPES = (
    "hybrid",
    "vggt_house_context",
    "vggt_house_full_tokens_nogate",
    "vggt_house_global_tokens_nogate",
    "vggt_house_global_embedding",
)

# Encoders carrying RGB at all, and hence able to back a `decoder=True` run.
# "cnn" belongs here but not above: its observation *is* the RGB image, so it
# needs no extraction.
RGB_BEARING_ENCODER_TYPES = ("cnn", *COMPOSITE_RGB_ENCODER_TYPES)
