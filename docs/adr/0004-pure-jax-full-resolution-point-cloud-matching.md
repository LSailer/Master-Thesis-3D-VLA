# Use pure-JAX spatial indexing for full-resolution point-cloud matching

Full-resolution VGGT point-cloud tracking will stay pure JAX and will not depend on SciPy/Open3D CPU trees in the tracking path. Because one full VGGT map has about 268k points, brute-force all-pairs nearest-neighbor matching is too memory- and compute-heavy; the implementation should use a JAX-native spatial index such as `jaxkd` first, with sorted voxel/spatial-hash matching as the fallback optimization if the KD-tree path is too slow.
