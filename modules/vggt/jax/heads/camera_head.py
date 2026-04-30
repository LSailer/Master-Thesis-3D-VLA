"""CameraHead — AdaLN-modulated iterative refiner for pose prediction.

Mirrors ``streamvggt.heads.camera_head.CameraHead`` for the **no-cache** path
(``use_cache=False`` in the reference). The cache branch lands in Step 6a+.

Each iteration:
    1. module_input = embed_pose(empty_pose_tokens | prev_pred_pose_enc)
    2. shift, scale, gate = split(SiLU; Linear)(module_input)
    3. pose_mod = gate * ((1 + scale) * adaln_norm(pose_tokens) + shift) + pose_tokens
    4. pose_mod = trunk[k](pose_mod, causal_mask) for k in 0..trunk_depth-1
    5. delta = pose_branch(trunk_norm(pose_mod))
    6. pred_pose_enc = pred_pose_enc + delta   (or delta on the first iter)
    7. append activate_pose(pred_pose_enc)

Uses explicit ``setup()`` rather than ``@nn.compact`` because the same
submodules (embed_pose, poseLN_modulation_1, trunk, pose_branch, trunk_norm)
are invoked once per iteration — Flax's compact mode forbids re-using a
name within a single ``__call__``.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from modules.vggt.jax.block import Block, Mlp


# Default activations for absT_quaR_FoV (CameraHead.__init__).
_TRANS_ACT = "linear"
_QUAT_ACT = "linear"
_FL_ACT = "relu"


def _inverse_log_transform(y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y) * jnp.expm1(jnp.abs(y))


def _base_pose_act(x: jnp.ndarray, act_type: str) -> jnp.ndarray:
    if act_type == "linear":
        return x
    if act_type == "inv_log":
        return _inverse_log_transform(x)
    if act_type == "exp":
        return jnp.exp(x)
    if act_type == "relu":
        return jax.nn.relu(x)
    raise ValueError(f"Unknown act_type: {act_type}")


def _activate_pose(
    pred_pose_enc: jnp.ndarray,
    trans_act: str = _TRANS_ACT,
    quat_act: str = _QUAT_ACT,
    fl_act: str = _FL_ACT,
) -> jnp.ndarray:
    T = _base_pose_act(pred_pose_enc[..., :3], trans_act)
    quat = _base_pose_act(pred_pose_enc[..., 3:7], quat_act)
    fl = _base_pose_act(pred_pose_enc[..., 7:], fl_act)
    return jnp.concatenate([T, quat, fl], axis=-1)


def _modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1 + scale) + shift


def _adaln_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """LayerNorm over last axis WITHOUT affine (elementwise_affine=False)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


class CameraHead(nn.Module):
    """Iterative camera-pose refiner over aggregator tokens (no cache)."""

    dim_in: int = 2048
    trunk_depth: int = 4
    target_dim: int = 9
    num_heads: int = 16
    mlp_ratio: float = 4.0
    init_values: float = 0.01
    norm_eps: float = 1e-5
    num_iterations: int = 4

    def setup(self):
        self.empty_pose_tokens = self.param(
            "empty_pose_tokens",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, 1, self.target_dim),
        )
        self.token_norm = nn.LayerNorm(epsilon=self.norm_eps, name="token_norm")
        self.embed_pose = nn.Dense(self.dim_in, use_bias=True, name="embed_pose")
        self.poseLN_modulation_1 = nn.Dense(
            3 * self.dim_in, use_bias=True, name="poseLN_modulation_1"
        )
        self.trunk_blocks = [
            Block(
                dim=self.dim_in,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=False,
                init_values=self.init_values,
                norm_eps=self.norm_eps,
                name=f"trunk_{k}",
            )
            for k in range(self.trunk_depth)
        ]
        self.trunk_norm = nn.LayerNorm(epsilon=self.norm_eps, name="trunk_norm")
        self.pose_branch = Mlp(
            hidden_features=self.dim_in // 2,
            out_features=self.target_dim,
            use_bias=True,
            name="pose_branch",
        )

    def __call__(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        *,
        use_cache: bool = False,
        past_kvs_camera: list | None = None,
    ):
        """Forward pass.

        Args:
            aggregated_tokens_list: Output of the aggregator — list of
                (B, S, P, C) tensors. Only the last element is used.
            use_cache: Streaming mode (S must be 1). Each trunk block's cache
                grows by ``num_iterations`` entries per frame.
            past_kvs_camera: List of length ``trunk_depth`` with per-block
                (k, v) tuples or None. Provided on every frame after the first.

        Returns:
            Without cache: ``pred_pose_enc_list`` (length = num_iterations).
            With cache:    ``(pred_pose_enc_list, new_past_kvs_camera)``.
        """
        tokens = aggregated_tokens_list[-1]
        B, S, P, C = tokens.shape
        if C != self.dim_in:
            raise ValueError(f"aggregator token dim {C} != dim_in {self.dim_in}")
        if use_cache and S != 1:
            raise ValueError(f"camera-head use_cache expects S=1, got S={S}")

        pose_tokens = self.token_norm(tokens[:, :, 0])  # (B, S, C)

        # Causal mask over S pose-token frames (no-cache only — under cache,
        # the rectangular Q vs [past_k | new_k] attention is naturally causal).
        if use_cache:
            attn_mask = None
        else:
            s_range = jnp.arange(S)
            future = s_range[:, None] < s_range[None, :]
            attn_mask = future.astype(jnp.float32) * jnp.finfo(jnp.float32).min

        if use_cache:
            if past_kvs_camera is None:
                past_kvs_camera = [None] * self.trunk_depth
            if len(past_kvs_camera) != self.trunk_depth:
                raise ValueError(
                    f"past_kvs_camera length {len(past_kvs_camera)} "
                    f"!= trunk_depth {self.trunk_depth}"
                )
            new_past_kvs_camera: list = list(past_kvs_camera)

        pred_pose_enc: jnp.ndarray | None = None
        pred_pose_enc_list: list[jnp.ndarray] = []

        for _ in range(self.num_iterations):
            if pred_pose_enc is None:
                pose_input = jnp.broadcast_to(
                    self.empty_pose_tokens.astype(pose_tokens.dtype),
                    (B, S, self.target_dim),
                )
            else:
                pose_input = jax.lax.stop_gradient(pred_pose_enc)

            module_input = self.embed_pose(pose_input)
            mod = self.poseLN_modulation_1(jax.nn.silu(module_input))
            shift_msa, scale_msa, gate_msa = jnp.split(mod, 3, axis=-1)

            pose_tokens_mod = (
                gate_msa * _modulate(_adaln_norm(pose_tokens), shift_msa, scale_msa)
                + pose_tokens
            )

            for k in range(self.trunk_depth):
                if use_cache:
                    pose_tokens_mod, new_kv = self.trunk_blocks[k](
                        pose_tokens_mod,
                        attn_mask=None,
                        past_kv=new_past_kvs_camera[k],
                        use_cache=True,
                    )
                    new_past_kvs_camera[k] = new_kv
                else:
                    pose_tokens_mod = self.trunk_blocks[k](
                        pose_tokens_mod, attn_mask=attn_mask
                    )

            delta = self.pose_branch(self.trunk_norm(pose_tokens_mod))
            pred_pose_enc = delta if pred_pose_enc is None else pred_pose_enc + delta
            pred_pose_enc_list.append(_activate_pose(pred_pose_enc))

        if use_cache:
            return pred_pose_enc_list, new_past_kvs_camera
        return pred_pose_enc_list
