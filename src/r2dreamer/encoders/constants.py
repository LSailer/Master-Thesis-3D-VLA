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

# Live house-points-pose sidecar. The accumulating VGGT house buffer grows every
# step, but jax.jit needs a static point count, so the adapter always emits a
# fixed-size `(HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM)` snapshot: the stored
# prefix zero-padded, plus the true count under HOUSE_CONTEXT_SIZE_KEY for
# masked mean/max pooling (HousePointsCameraEncoder._house_embedding). Sized to
# hold a full 1 cm map (~210k voxels measured after 50 steps on one L1 scene);
# the branch costs ~0.9 ms fwd / ~2.3 ms fwd+bwd on H100 at this size
# (bench_house_branch_sizes.py, job 5695204).
HOUSE_POINT_DIM = 6  # xyz(3) + rgb(3), rgb normalized to [0, 1]
HOUSE_CONTEXT_MAX_POINTS = 262_144
