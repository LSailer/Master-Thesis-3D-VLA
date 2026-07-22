"""Device-side point-cloud geometry ops (pure JAX, no Open3D).

Counterpart to ``src.shared.ply_io``: transforms here run on the JAX
device with array inputs/outputs; ``ply_io`` marks the device -> host
boundary for file writes. Keep Open3D out of this module so device code
paths never import it.
"""

import jax
import jax.numpy as jnp


def voxel_down_sample(
    xyz: jnp.ndarray, rgb: jnp.ndarray, voxel_size: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Voxel-downsample points with per-voxel mean reduction, on the JAX device.

    Replicates ``open3d.t.geometry.PointCloud.voxel_down_sample`` (floor to
    voxel grid, average positions/colors per occupied voxel) without Open3D:
    the CUDA build of that op segfaults on this cluster even in a pure
    Open3D process, so downsampling stays in JAX and Open3D is only used
    for host-side PLY I/O. The grid is anchored at the world origin
    (``floor(xyz / voxel_size)``), matching the persistent voxel keys in
    ``src.buffer.house_context_pose_buffer``, not at the cloud's min bound
    as in Open3D — cells are shifted by a sub-voxel offset.

    Not jittable: the number of occupied voxels is data-dependent, so the
    output shape is dynamic. Intended for one-shot cold paths.

    Args:
      xyz: ``(N, 3)`` float point positions.
      rgb: ``(N, 3)`` float point colors in ``[0, 1]``.
      voxel_size: Voxel edge length in meters; must be positive.

    Returns:
      Tuple ``(xyz_down, rgb_down)`` of ``(M, 3)`` float32 arrays, one row
      per occupied voxel, each the mean of that voxel's members.

    Raises:
      ValueError: If ``voxel_size`` is not positive.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    # float32 before the divide: bfloat16 resolution at meter-scale coords is
    # coarser than the voxel edge and would corrupt voxel assignment.
    xyz = xyz.astype(jnp.float32)
    rgb = rgb.astype(jnp.float32)
    # Row-wise unique instead of bit-packed keys: without jax_enable_x64,
    # int64 silently degrades to int32 and shifting >=32 bits corrupts keys.
    coords = jnp.floor(xyz / voxel_size).astype(jnp.int32)
    uniq, inverse = jnp.unique(coords, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    num_voxels = uniq.shape[0]
    counts = jax.ops.segment_sum(
        jnp.ones((xyz.shape[0],), dtype=jnp.float32), inverse, num_voxels
    )
    xyz_down = jax.ops.segment_sum(xyz, inverse, num_voxels) / counts[:, None]
    rgb_down = jax.ops.segment_sum(rgb, inverse, num_voxels) / counts[:, None]
    return xyz_down, rgb_down
