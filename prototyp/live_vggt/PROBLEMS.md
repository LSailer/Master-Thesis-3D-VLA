# Open problems / research notes

- **bfloat16 re-seeding drift in `GrowableHouseContextBuffer`.** The backing
  store keeps representatives in bfloat16; growth re-seeds those *rounded*
  positions into the doubled buffer. Where bf16 spacing (~|coord| / 128)
  approaches `voxel_size_m`, neighbouring voxels merge and re-added frames can
  leave near-duplicate representatives one voxel apart. At house scale
  (|xyz| ~ 10 m, 1 cm voxels) displacement is ~3 cm — fine for visualization,
  not for exact voxel accounting. Surfaced by the first version of
  `tests/prototyp/live_vggt/test_growable_buffer.py` (offsets at x≈1000 m
  collapsed 90 points to 68).
