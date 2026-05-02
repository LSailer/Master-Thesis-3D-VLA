"""Verification spike for blocker B4: print actual shape of aggregated_tokens.

The encoder-fusion plan assumes 37x37x1024 patch tokens; this script confirms
the real layout (likely (B, S, prefix + 1369, C) given patch_start_idx).

Run on H100:
    srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00 --pty \\
        uv run python scripts/verify_aggregated_tokens.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from modules.vggt.feature_extractor import VGGTFeatureExtractor  # noqa: E402


def main() -> None:
    ext = VGGTFeatureExtractor(device="cuda", compile=False)

    captured: dict = {}
    original = ext.model.aggregator

    def spy(images, **kwargs):
        out = original(images, **kwargs)
        agg, patch_start_idx, _ = out
        captured["agg_type"] = type(agg).__name__
        if isinstance(agg, (list, tuple)):
            captured["agg_len"] = len(agg)
            captured["agg_per_layer"] = [
                (tuple(t.shape), str(t.dtype), str(t.device)) for t in agg
            ]
        else:
            captured["agg_shape"] = tuple(agg.shape)
            captured["agg_dtype"] = str(agg.dtype)
            captured["agg_device"] = str(agg.device)
        captured["patch_start_idx"] = int(patch_start_idx)
        return out

    # Bypass nn.Module.__setattr__ which rejects non-Module callables.
    object.__setattr__(ext.model, "aggregator", spy)

    rgb = (np.random.rand(3, 518, 518) * 255).astype(np.uint8)
    out = ext.extract(rgb)

    print("=" * 60)
    print("aggregated_tokens introspection")
    print("=" * 60)
    for k, v in captured.items():
        print(f"  {k}: {v}")

    print()
    print("extract() outputs:")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    print()
    if "agg_shape" in captured:
        s = captured["agg_shape"]
        psi = captured["patch_start_idx"]
        if len(s) >= 3:
            n_patches = s[-2] - psi if s[-2] > psi else s[-2]
            print(f"Inferred patch tokens: {n_patches}")
            print(f"Expected for 37x37 grid: 1369")
            if n_patches == 1369:
                ch = s[-1]
                print(f"OK — patch tokens are {n_patches} x {ch}, can reshape to (37, 37, {ch})")
            else:
                print("WARNING — patch count != 1369; plan needs revision")


if __name__ == "__main__":
    with torch.no_grad():
        main()
