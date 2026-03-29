"""VGGT variant loading and registry."""

import torch
import importlib
from typing import Optional


VARIANTS = {
    "vggt": {
        "repo": "facebookresearch/vggt",
        "module": "vggt.models.vggt",
        "class": "VGGT",
    },
    "stream_vggt": {
        "repo": "wzzheng/StreamVGGT",
        "module": "stream_vggt.models.stream_vggt",
        "class": "StreamVGGT",
    },
    "infinite_vggt": {
        "repo": "AutoLab-SAI-SJTU/InfiniteVGGT",
        "module": "infinite_vggt.models.infinite_vggt",
        "class": "InfiniteVGGT",
    },
}


def get_available_variants() -> list[str]:
    """Return names of variants whose modules can be imported."""
    available = []
    for name, info in VARIANTS.items():
        try:
            importlib.import_module(info["module"])
            available.append(name)
        except ImportError:
            continue
    return available


def load_variant(name: str, device: Optional[str] = None) -> torch.nn.Module:
    """Load a VGGT variant by name, returning a frozen model on the given device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    info = VARIANTS[name]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    model = cls.from_pretrained()
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
