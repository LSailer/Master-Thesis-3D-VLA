# Open3D 0.19: CPU-only or GPU/CUDA? (primary-source findings)

**TL;DR.** The claim "Open3D is a host-side CPU-only library" is **partly wrong for
Open3D ≥0.17 on Linux x86_64**. The default `pip install open3d` wheel on Linux
x86_64 *is CUDA-enabled*: the tensor API (`open3d.t.geometry`) runs kernels (incl.
`voxel_down_sample`) on `CUDA:0` when the point cloud lives on a CUDA device. The
legacy `open3d.geometry.PointCloud` API is CPU-only. macOS/Windows wheels are
CPU-only. File I/O (`write_point_cloud`) is inherently host-side. JAX zero-copy
interop is *technically possible* via DLPack but is **not officially documented or
tested** (only PyTorch + NumPy are).

## Q1 — Does `pip install open3d` ship CUDA? Which platforms?
**Yes, on Linux x86_64.** The 0.19.0 default wheel
`open3d-0.19.0-cp311-cp311-manylinux_2_31_x86_64.whl` is **447.7 MB**, whereas the
separate `open3d-cpu-0.19.0-...manylinux...whl` is **99.7 MB** (verified via
`pypi.org/pypi/open3d/0.19.0/json` and `.../open3d-cpu/0.19.0/json`, 2026-07-21).
The ~4.5× size gap is the bundled CUDA runtime/kernels. Docs confirm the CPU wheel
is a deliberate alternative: *"pip install open3d-cpu — Smaller CPU only wheel on
x86_64 Linux (since v0.17+)"* — the existence of an explicit `-cpu` variant since
v0.17 implies the plain Linux wheel is the CUDA build (getting_started, docs/release).
The docs pip-wheel table lists the Linux binaries as *"x86_64 (CXX11 ABI) with
CUDA 11.x"*. macOS (`macosx_..._universal2`) and Windows (`win_amd64`) wheels carry
no CUDA. No `aarch64`/ARM64 CUDA wheel ships. Building your own CUDA wheel uses
`cmake -DBUILD_CUDA_MODULE=ON` + `make install-pip-package` (compilation.html);
`BUILD_CUDA_MODULE` defaults OFF.
*Discrepancy:* the v0.19 blog announces *"CUDA 12 support"* while the docs table
still says *"CUDA 11.x"* — see Open questions.

## Q2 — Which APIs use the GPU? Is `voxel_down_sample` GPU-accelerated?
The **tensor API** (`open3d.t.geometry.PointCloud`) is device-aware: it has
`.cuda()`, `.cpu()`, `.to(device)`; *"The device of 'positions' determines the
device of the point cloud"* (python_api/open3d.t.geometry.PointCloud). A maintainer
confirms in isl-org/Open3D#5519: *"You should create your point cloud on CUDA
device, then the voxel down sample will be performed on GPU automatically"* (check
`o3d.core.cuda.is_available()`, read via `open3d.t.io.read_point_cloud`, then
`pcd.cuda()`). So **`voxel_down_sample` is GPU-accelerated in the tensor API**.
The **legacy** `open3d.geometry.PointCloud` has no device concept and runs on CPU
only. (Caveat from #5519: GPU voxel downsample can OOM on very large clouds, e.g.
>112M points.)

## Q3 — Is file I/O (`write_point_cloud`) GPU-accelerated?
**No — inherently host-side.** Serialization to PLY/PCD/etc. is CPU/disk work; a
GPU-resident tensor point cloud must be copied to host before/while writing. This
holds for both `open3d.io.write_point_cloud` (legacy) and `open3d.t.io`. No primary
source claims GPU file I/O; the tensor I/O tutorials read into a device tensor but
the write path is host serialization.

## Q4 — Zero-copy interop / DLPack (PyTorch, CuPy, JAX)?
The tensor API supports **DLPack** via `o3c.Tensor.from_dlpack()` / `.to_dlpack()`;
the returned tensor **shares memory** (zero-copy), incl. on CUDA. The official
tensor tutorial documents **only PyTorch and NumPy** interop (NumPy uses
`from_numpy()`/`numpy()`, host-only). **JAX is not mentioned anywhere**, nor CuPy.
Because Open3D's `from_dlpack` accepts any object exposing `__dlpack__` (which JAX
arrays implement), a JAX CUDA array can *in principle* be handed to Open3D
zero-copy, but this is **undocumented and untested** by Open3D, and comes with real
caveats: JAX arrays are immutable and manage their own CUDA memory/streams, with no
cross-framework stream-synchronization guarantee, and both stacks must share a
compatible CUDA runtime.

## Corrections to the claim
Original claim: *"Open3D is a host-side C++ library — a JAX training loop can't use
it on-device per step; data must cross device→host."*

- **Wrong** that Open3D is CPU/host-only: the default Linux x86_64 PyPI wheel is
  CUDA-enabled and the `open3d.t` tensor API runs point-cloud ops (incl.
  `voxel_down_sample`) on-GPU. Correct the blanket "host-side CPU library" wording.
- **Correct** that the *legacy* `open3d.geometry` API is CPU-only — if the code uses
  that API, the device→host copy is real.
- **Correct** for **file I/O**: `write_point_cloud` is host-side regardless of API.
- **Mostly correct in practice for a JAX loop specifically:** there is no official
  JAX↔Open3D zero-copy path; the DLPack bridge exists but is undocumented/untested
  for JAX, so a robust per-step integration realistically still copies device→host
  (or requires unsupported DLPack plumbing). State it as "no *supported* on-device
  JAX path," not "Open3D can't run on GPU."

## Sources
- PyPI wheel metadata (sizes/filenames): https://pypi.org/pypi/open3d/0.19.0/json ,
  https://pypi.org/pypi/open3d-cpu/0.19.0/json (fetched 2026-07-21)
- Getting started (open3d-cpu since v0.17, CUDA 11.x Linux wheels):
  https://www.open3d.org/docs/release/getting_started.html
- v0.19 release blog (CUDA 12 support, SYCL preview):
  https://www.open3d.org/2025/01/09/open3d-v0-19-is-out-with-new-features-and-more-gpu-support/
- Build from source (`BUILD_CUDA_MODULE`): https://www.open3d.org/docs/release/compilation.html
- Tensor core tutorial (Device CUDA:0, DLPack PyTorch/NumPy):
  https://www.open3d.org/docs/release/tutorial/core/tensor.html
- t.geometry.PointCloud API (cuda()/cpu()/to()):
  https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html
- Maintainer confirmation, GPU voxel_down_sample: https://github.com/isl-org/Open3D/issues/5519

## Open questions
- Which CUDA does the *shipped* 0.19 Linux wheel actually link — 11.x (docs table)
  or 12 (blog "CUDA 12 support")? The blog may describe build capability, not the
  released binary. Would need to inspect the wheel's bundled `.so`/CUDA libs to settle.
- Exact first version with CUDA-enabled PyPI wheels — docs only pin the `-cpu`
  split at v0.17+; CUDA Linux wheels likely predate that but not confirmed here.
- Whether any real JAX→Open3D `from_dlpack` transfer works without a host round-trip
  on matching CUDA runtimes — not tested; no primary source either way.
