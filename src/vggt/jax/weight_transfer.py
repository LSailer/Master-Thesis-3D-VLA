"""PyTorch StreamVGGT state_dict -> JAX/Flax PyTree.

Produces a nested dict whose leaves are numpy arrays in Flax layout:
  - Conv2d         (O, I, H, W)    -> kernel (H, W, I, O)
  - ConvTranspose2d(I, O, H, W)    -> kernel (H, W, I, O)
  - Linear         (O, I)          -> kernel (I, O)
  - LayerNorm weight               -> scale  (unchanged)
  - Bias / param tensors           -> unchanged

The returned tree is consumed directly by ``flax_module.apply({"params": tree}, ...)``.
Step 1 only produces and validates this tree; the Flax modules that consume
it land in step 2+.

v1 scope excludes ``depth_head.*`` and ``track_head.*`` -- the plan ports
only aggregator, camera_head, and point_head.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.vggt.paths import infinite_vggt_src

# --------------------------------------------------------------------------- #
#  Scope
# --------------------------------------------------------------------------- #

V1_EXCLUDE_PREFIXES: tuple[str, ...] = ("depth_head.", "track_head.")
"""State_dict prefixes that are ignored in v1 (per plan non-goals)."""


# --------------------------------------------------------------------------- #
#  Transposition helpers
# --------------------------------------------------------------------------- #


def _conv2d_to_flax(t: np.ndarray) -> np.ndarray:
    """(O, I, H, W) -> (H, W, I, O)."""
    assert t.ndim == 4, t.shape
    return np.transpose(t, (2, 3, 1, 0))


def _conv_transpose2d_to_flax(t: np.ndarray) -> np.ndarray:
    """(I, O, H, W) -> (H, W, I, O) with spatial axes reversed.

    Flax ``nn.ConvTranspose`` (default ``transpose_kernel=False``) expects
    kernel shape ``(*spatial, in_features, features)`` — same layout as
    ``nn.Conv``. However the numerics of JAX's ``lax.conv_transpose`` treat
    the kernel as the FORWARD-conv kernel whose transpose it computes, so
    the kernel must be spatially flipped (H and W reversed) relative to
    PyTorch's ``ConvTranspose2d`` convention.

    Verified empirically: for a minimal 5x5 -> 20x20 conv_transpose with
    stride=4, ``np.flip(np.transpose(pt_w, (2, 3, 0, 1)), axis=(0, 1))``
    produces bit-exact parity with PyTorch while the un-flipped form gives
    ~10-15 max-abs error.
    """
    assert t.ndim == 4, t.shape
    jax_layout = np.transpose(t, (2, 3, 0, 1))  # (I, O, H, W) -> (H, W, I, O)
    return np.flip(jax_layout, axis=(0, 1)).copy()


def _linear_to_flax(t: np.ndarray) -> np.ndarray:
    """(O, I) -> (I, O)."""
    assert t.ndim == 2, t.shape
    return np.transpose(t, (1, 0))


# --------------------------------------------------------------------------- #
#  Key -> path / leaf-name rewriting
# --------------------------------------------------------------------------- #

# Each rule is (regex, path_template, leaf_name_map, transpose).
#   - path_template: dotted output path; use \1, \2 for captured groups and
#     ``_N`` markers are replaced by capture groups in order.
#   - leaf_name_map: dict mapping the last component of the PyTorch key
#     ("weight", "bias", "gamma", or a literal param name) to the output leaf
#     name ("kernel", "scale", "bias", "gamma", or the literal).
#   - transpose: one of None, "conv", "conv_transpose", "linear".
#
# The rules are ordered; the first match wins. Patterns are anchored with ^/$.

_Rule = tuple[re.Pattern[str], str, dict[str, str], str | None]


def _rule(
    pattern: str,
    path_template: str,
    leaf_map: dict[str, str],
    transpose: str | None = None,
) -> _Rule:
    return re.compile("^" + pattern + "$"), path_template, leaf_map, transpose


# Leaf maps
_NORM_MAP = {"weight": "scale", "bias": "bias"}
_CONV_MAP = {"weight": "kernel", "bias": "bias"}
_LINEAR_MAP = {"weight": "kernel", "bias": "bias"}
_LS_MAP = {"gamma": "gamma"}

# --------------------------------------------------------------------------- #
#  Rule builders for the repeated block families
# --------------------------------------------------------------------------- #


def _make_attention_rules(
    prefix: str, out_prefix: str, has_qk_norm: bool
) -> list[_Rule]:
    """Rules for a transformer-block attention submodule."""
    rules = [
        _rule(
            rf"{prefix}\.(\d+)\.attn\.qkv\.(weight|bias)",
            rf"{out_prefix}_\1.attn.qkv",
            _LINEAR_MAP,
            "linear",
        ),
        _rule(
            rf"{prefix}\.(\d+)\.attn\.proj\.(weight|bias)",
            rf"{out_prefix}_\1.attn.proj",
            _LINEAR_MAP,
            "linear",
        ),
    ]
    if has_qk_norm:
        rules.extend(
            [
                _rule(
                    rf"{prefix}\.(\d+)\.attn\.(q_norm|k_norm)\.(weight|bias)",
                    rf"{out_prefix}_\1.attn.\2",
                    _NORM_MAP,
                    None,
                ),
            ]
        )
    return rules


def _make_block_rules(prefix: str, out_prefix: str, has_qk_norm: bool) -> list[_Rule]:
    """Rules for a full transformer Block (norm1 + attn + ls1 + norm2 + mlp + ls2)."""
    return [
        _rule(
            rf"{prefix}\.(\d+)\.norm1\.(weight|bias)",
            rf"{out_prefix}_\1.norm1",
            _NORM_MAP,
            None,
        ),
        _rule(
            rf"{prefix}\.(\d+)\.norm2\.(weight|bias)",
            rf"{out_prefix}_\1.norm2",
            _NORM_MAP,
            None,
        ),
        _rule(
            rf"{prefix}\.(\d+)\.ls1\.(gamma)",
            rf"{out_prefix}_\1.ls1",
            _LS_MAP,
            None,
        ),
        _rule(
            rf"{prefix}\.(\d+)\.ls2\.(gamma)",
            rf"{out_prefix}_\1.ls2",
            _LS_MAP,
            None,
        ),
        _rule(
            rf"{prefix}\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)",
            rf"{out_prefix}_\1.mlp.\2",
            _LINEAR_MAP,
            "linear",
        ),
        *_make_attention_rules(
            prefix=rf"{prefix}",
            out_prefix=f"{out_prefix}",
            has_qk_norm=has_qk_norm,
        ),
    ]


# The functions below are forward-referenced in RULES; we build them as
# plain lists using the helpers above. Defined at module scope after RULES
# via a closure trick is awkward, so we defer by making them module-level
# builders that RULES looks up. Instead, we just define them before use --
# but RULES is computed at import, so we need to define these first.


def _dinov2_block_rules(prefix: str, out_prefix: str, block_name: str) -> list[_Rule]:
    # DINOv2 blocks in the reference have no QK norm (qk_norm defaults to False
    # for vit_large) -- no k_norm / q_norm keys appear in the checkpoint under
    # `aggregator.patch_embed.blocks.*`.
    return _make_block_rules(
        prefix=prefix,
        out_prefix=f"{out_prefix}.{block_name}",
        has_qk_norm=False,
    )


def _aggregator_block_rules(variant: str) -> list[_Rule]:
    # Aggregator frame_blocks / global_blocks use qk_norm=True.
    assert variant in ("frame", "global")
    return _make_block_rules(
        prefix=rf"aggregator\.{variant}_blocks",
        out_prefix=f"aggregator.{variant}_blocks",
        has_qk_norm=True,
    )


def _camera_trunk_rules() -> list[_Rule]:
    # Camera head trunk blocks: no qk_norm, no rope, dim=2048, num_heads=16.
    return _make_block_rules(
        prefix=r"camera_head\.trunk",
        out_prefix="camera_head.trunk",
        has_qk_norm=False,
    )


# Forward-referenced builders are used in RULES; rebuild RULES now that they
# exist (Python order of definition means we need to redo the construction).
RULES = [
    _rule(r"aggregator\.(camera_token|register_token)", r"aggregator", {}, None),
    _rule(
        r"aggregator\.patch_embed\.(cls_token|mask_token|register_tokens|pos_embed)",
        r"aggregator.patch_embed",
        {},
        None,
    ),
    _rule(
        r"aggregator\.patch_embed\.norm\.(weight|bias)",
        r"aggregator.patch_embed.norm",
        _NORM_MAP,
        None,
    ),
    _rule(
        r"aggregator\.patch_embed\.patch_embed\.proj\.(weight|bias)",
        r"aggregator.patch_embed.patch_embed.proj",
        _CONV_MAP,
        "conv",
    ),
    *_dinov2_block_rules(
        prefix=r"aggregator\.patch_embed\.blocks",
        out_prefix="aggregator.patch_embed",
        block_name="blocks",
    ),
    *_aggregator_block_rules(variant="frame"),
    *_aggregator_block_rules(variant="global"),
    _rule(r"camera_head\.empty_pose_tokens", r"camera_head", {}, None),
    _rule(
        r"camera_head\.embed_pose\.(weight|bias)",
        r"camera_head.embed_pose",
        _LINEAR_MAP,
        "linear",
    ),
    _rule(
        r"camera_head\.token_norm\.(weight|bias)",
        r"camera_head.token_norm",
        _NORM_MAP,
        None,
    ),
    _rule(
        r"camera_head\.trunk_norm\.(weight|bias)",
        r"camera_head.trunk_norm",
        _NORM_MAP,
        None,
    ),
    _rule(
        r"camera_head\.adaln_norm\.(weight|bias)",
        r"camera_head.adaln_norm",
        _NORM_MAP,
        None,
    ),
    _rule(
        r"camera_head\.poseLN_modulation\.1\.(weight|bias)",
        r"camera_head.poseLN_modulation_1",
        _LINEAR_MAP,
        "linear",
    ),
    _rule(
        r"camera_head\.pose_branch\.(fc1|fc2)\.(weight|bias)",
        r"camera_head.pose_branch.\1",
        _LINEAR_MAP,
        "linear",
    ),
    *_camera_trunk_rules(),
    _rule(r"point_head\.norm\.(weight|bias)", r"point_head.norm", _NORM_MAP, None),
    _rule(
        r"point_head\.projects\.(\d+)\.(weight|bias)",
        r"point_head.projects_\1",
        _CONV_MAP,
        "conv",
    ),
    _rule(
        r"point_head\.resize_layers\.(0|1)\.(weight|bias)",
        r"point_head.resize_layers_\1",
        _CONV_MAP,
        "conv_transpose",
    ),
    _rule(
        r"point_head\.resize_layers\.(3)\.(weight|bias)",
        r"point_head.resize_layers_\1",
        _CONV_MAP,
        "conv",
    ),
    _rule(
        r"point_head\.scratch\.(layer[1-4]_rn)\.(weight|bias)",
        r"point_head.scratch.\1",
        _CONV_MAP,
        "conv",
    ),
    _rule(
        r"point_head\.scratch\.(refinenet[1-4])\.out_conv\.(weight|bias)",
        r"point_head.scratch.\1.out_conv",
        _CONV_MAP,
        "conv",
    ),
    _rule(
        r"point_head\.scratch\.(refinenet[1-4])\.(resConfUnit[12])\.(conv[12])\.(weight|bias)",
        r"point_head.scratch.\1.\2.\3",
        _CONV_MAP,
        "conv",
    ),
    _rule(
        r"point_head\.scratch\.output_conv1\.(weight|bias)",
        r"point_head.scratch.output_conv1",
        _CONV_MAP,
        "conv",
    ),
    _rule(
        r"point_head\.scratch\.output_conv2\.(0|2)\.(weight|bias)",
        r"point_head.scratch.output_conv2_\1",
        _CONV_MAP,
        "conv",
    ),
]


# --------------------------------------------------------------------------- #
#  Translation
# --------------------------------------------------------------------------- #

_PARAM_LEAF_NAMES = (
    "camera_token",
    "register_token",
    "cls_token",
    "mask_token",
    "register_tokens",
    "pos_embed",
    "empty_pose_tokens",
)


def _split_key_suffix(key: str) -> tuple[str, str]:
    """Split into (prefix, last_component). Last component is e.g. "weight", "bias",
    "gamma", or a top-level parameter name like "cls_token"."""
    idx = key.rfind(".")
    return (key[:idx], key[idx + 1 :]) if idx >= 0 else ("", key)


def _apply_transpose(t: np.ndarray, kind: str | None, leaf_name: str) -> np.ndarray:
    """Dispatch transposition. Biases / scales / gammas / raw params pass through."""
    if kind is None or leaf_name in ("bias", "scale", "gamma"):
        return t
    if leaf_name == "kernel":
        if kind == "conv":
            return _conv2d_to_flax(t) if t.ndim == 4 else _linear_to_flax(t)
        if kind == "conv_transpose":
            return _conv_transpose2d_to_flax(t)
        if kind == "linear":
            return _linear_to_flax(t)
    return t


def _match_rule(key: str) -> tuple[_Rule, re.Match[str]] | None:
    for rule in RULES:
        pat, _, _, _ = rule
        m = pat.match(key)
        if m is not None:
            return rule, m
    return None


def _resolve_leaf_name(key: str, leaf_map: dict[str, str]) -> str:
    """Determine the JAX leaf name from the PyTorch key's last component."""
    _, last = _split_key_suffix(key)
    if last in _PARAM_LEAF_NAMES:
        return last
    if last in leaf_map:
        return leaf_map[last]
    # Fallback: raw parameter name preserved (e.g. camera_token without suffix)
    return last


def _insert(tree: dict[str, Any], path: str, leaf: str, value: np.ndarray) -> None:
    """Insert ``value`` at ``tree[path][leaf]``, creating intermediate dicts."""
    node = tree
    if path:
        for part in path.split("."):
            node = node.setdefault(part, {})
    if leaf in node:
        raise KeyError(f"Duplicate destination: {path}/{leaf}")
    node[leaf] = value


def _expand_path_template(key: str, path_template: str, match: re.Match[str]) -> str:
    """Expand a matched output path template for one PyTorch key."""
    try:
        return match.expand(path_template)
    except re.error as e:
        raise RuntimeError(f"Bad path_template for key {key!r}: {e}") from e


def _map_state_dict_entry(
    key: str, tensor: np.ndarray
) -> tuple[str, str, np.ndarray] | None:
    """Map one PyTorch state_dict entry to (Flax path, leaf name, value)."""
    matched = _match_rule(key)
    if matched is None:
        return None

    (_pat, path_template, leaf_map, transpose_kind), match = matched
    out_path = _expand_path_template(key, path_template, match)
    leaf_name = _resolve_leaf_name(key, leaf_map)
    value = _apply_transpose(np.asarray(tensor), transpose_kind, leaf_name)
    return out_path, leaf_name, value


def load_pytorch_weights(
    state_dict: dict[str, np.ndarray],
    *,
    include_v1_only: bool = True,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Convert a PyTorch StreamVGGT ``state_dict`` to a JAX PyTree.

    Args:
        state_dict: mapping from PyTorch parameter names to numpy arrays.
        include_v1_only: if True (default), skip depth_head and track_head.

    Returns:
        (params_tree, report) where ``report`` has keys:
          - ``"mapped"``:    list of state_dict keys that were mapped.
          - ``"skipped"``:   list of state_dict keys skipped by v1 scope.
          - ``"unmapped"``:  list of state_dict keys that no rule matched
                             (this MUST be empty for Level-1 to pass).

    Raises:
        No exceptions on rule miss -- the caller checks ``report["unmapped"]``.
    """
    tree: dict[str, Any] = {}
    report: dict[str, list[str]] = {"mapped": [], "skipped": [], "unmapped": []}

    for key, tensor in state_dict.items():
        if include_v1_only and key.startswith(V1_EXCLUDE_PREFIXES):
            report["skipped"].append(key)
            continue

        mapped = _map_state_dict_entry(key, tensor)
        if mapped is None:
            report["unmapped"].append(key)
            continue

        out_path, leaf_name, value = mapped
        _insert(tree, out_path, leaf_name, value)
        report["mapped"].append(key)

    return tree, report


# --------------------------------------------------------------------------- #
#  Coverage / round-trip checks (Level-1 exit criterion)
# --------------------------------------------------------------------------- #


def verify_coverage(
    state_dict: dict[str, np.ndarray],
    report: dict[str, list[str]],
) -> None:
    """Assert every v1-scope state_dict key was mapped exactly once.

    Raises AssertionError if any key is unmapped.
    """
    unmapped = report.get("unmapped", [])
    assert not unmapped, f"{len(unmapped)} unmapped keys; first 10: {unmapped[:10]}"

    in_scope = [k for k in state_dict if not k.startswith(V1_EXCLUDE_PREFIXES)]
    mapped = set(report.get("mapped", []))
    missing = [k for k in in_scope if k not in mapped]
    assert not missing, (
        f"{len(missing)} in-scope keys not mapped; first 10: {missing[:10]}"
    )


def verify_per_leaf_roundtrip(
    state_dict: dict[str, np.ndarray],
    tree: dict[str, Any],
) -> None:
    """Assert every JAX leaf equals the transposed source tensor exactly (atol=0).

    Iterates state_dict rather than the tree so we also catch duplicate mappings.
    """
    for key, tensor in state_dict.items():
        if key.startswith(V1_EXCLUDE_PREFIXES):
            continue
        mapped = _map_state_dict_entry(key, tensor)
        assert mapped is not None, f"Unmapped: {key}"
        out_path, leaf_name, expected = mapped

        node = tree
        if out_path:
            for part in out_path.split("."):
                node = node[part]
        got = node[leaf_name]
        if got.shape != expected.shape:
            raise AssertionError(
                f"Shape mismatch at {out_path}/{leaf_name}: got {got.shape}, expected {expected.shape} (from {key})"
            )
        if not np.array_equal(got, expected):
            raise AssertionError(
                f"Value mismatch at {out_path}/{leaf_name} (from {key})"
            )


def count_leaves(tree: dict[str, Any]) -> int:
    n = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        for v in node.values():
            if isinstance(v, dict):
                stack.append(v)
            else:
                n += 1
    return n


def sum_numel(tree: dict[str, Any]) -> int:
    s = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        for v in node.values():
            if isinstance(v, dict):
                stack.append(v)
            else:
                s += int(np.asarray(v).size)
    return s


# --------------------------------------------------------------------------- #
#  Checkpoint loading (HuggingFace)
# --------------------------------------------------------------------------- #

_DEFAULT_REPO = "lch01/StreamVGGT"


def load_checkpoint(repo: str = _DEFAULT_REPO) -> dict[str, np.ndarray]:
    """Download (or hit HF cache) and return StreamVGGT weights as numpy arrays.

    Keeps the dependency on the PyTorch StreamVGGT class for one-time loading
    (the checkpoint is pickled as a torch state_dict via PyTorchModelHubMixin),
    then detaches into numpy so downstream code never needs torch again.
    """
    import sys

    ivggt_src = infinite_vggt_src()
    if str(ivggt_src) not in sys.path:
        sys.path.insert(0, str(ivggt_src))

    import torch  # noqa: E402 -- deferred until used
    from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402

    with torch.device("cpu"):
        model = StreamVGGT.from_pretrained(repo)
    sd = model.state_dict()
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}
