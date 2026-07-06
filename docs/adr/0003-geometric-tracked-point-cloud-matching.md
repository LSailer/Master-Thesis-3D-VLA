# Match TrackedPointCloud points geometrically, not index-wise

`TrackedPointCloud` will decide whether a current VGGT point is already known by geometric proximity to the global scene cloud, not by exact index-wise equality with the previous frame. Full-resolution VGGT point maps can shift per pixel between observations, so pixel index stability is not a valid identity contract; visibility should be assigned to an existing global point when a nearest-neighbor or voxel-radius match is within the configured threshold, otherwise a new tracked point is appended.
