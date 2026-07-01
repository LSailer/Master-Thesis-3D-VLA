"""Shared dimensions for R2Dreamer encoder modules."""

# Hybrid encoder legacy packed input layout. Current replay/live paths keep
# modalities under explicit fields and HybridEncoder owns any packing needed.
HYBRID_RGB_DIM = 3 * 64 * 64  # 12288 — the CNN branch's 64x64 RGB, flattened
HYBRID_VGGT_DIM = 4116  # WP/CP width: world_points 37*37*3 + camera_pose 9

# Aggregator readouts (3D-50 follow-up). Defined here (small ints) rather than
# imported from adapters.vggt_adapter so importing encoder modules stays free of
# the heavy VGGT extractor dependency; tests assert these agree with the adapter
# constants and the live extractor shape.
AGG_RAW_DIM = 1370 * 1024  # cam + patches, 4 register tokens dropped
AGG_TOKEN_TOKENS = 1374  # cam + registers + patches
HOUSE_CONTEXT_DIM = 1024
AGG_REGISTER_TOKENS = 4
