"""Canonical observation field names shared by adapters and encoders."""

HYBRID_IMAGE_KEY = "image"
HYBRID_WP_CP_KEY = "wp_cp"
HOUSE_CONTEXT_KEY = "house_context"
HOUSE_CONTEXT_SIZE_KEY = "house_context_size"  # valid-row count for masked pooling
FULL_TOKENS_KEY = "full_tokens"
GLOBAL_TOKENS_KEY = "global_tokens"
# House-global-embedding L1 variant: the global-half aggregator tokens split
# into the camera token (Position Signal, [0:1]) and the patch tokens ([5:],
# dropping the 4 register tokens) — see src/prototyp/house_global_embedding/
# IDEA.md. Stored as separate replay fields so the PointNet reducer pools only
# the patches and keeps the camera token on its own side branch.
CAMERA_TOKEN_GLOBAL_KEY = "camera_token_global"
GLOBAL_PATCH_TOKENS_KEY = "global_patch_tokens"
WORLD_POINTS_KEY = "world_points"
CAMERA_POSE_KEY = "camera_pose"
