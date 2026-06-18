"""Pure-NumPy metrics for the VGGT first-frame invariance probe.

VGGT predicts World Points (WP) and Camera Pose (CP) relative to the *first frame*
of its streaming window. Two runs over the same physical path but with different
first frames therefore express the same geometry in different reference frames.
These helpers quantify three things between a forward and a backward run:

  1. Raw divergence of what the Dreamer model actually ingests (WP/CP vectors).
  2. How much of that divergence is a benign rigid reference-frame transform
     (Umeyama alignment residual) vs. a genuine structural difference.
  3. Divergence of the trained Dreamer RSSM posterior (categorical JS + deter cos).

No torch/jax — safe to run on the login node / inside the CPU analysis notebook.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Vector similarity
# --------------------------------------------------------------------------- #
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.ravel() - b.ravel()))


# --------------------------------------------------------------------------- #
# Rigid alignment (Umeyama / Kabsch) — separates reference-frame change from
# genuine geometric difference.
# --------------------------------------------------------------------------- #
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Best similarity transform mapping src -> dst (least squares).

    src, dst : (N, 3) point sets in correspondence.
    Returns dict with R (3,3), t (3,), s (float), aligned (N,3),
    rmse_before (raw src vs dst) and rmse_after (aligned vs dst).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd.T @ xs) / n
    u, d_diag, vt = np.linalg.svd(cov)
    sgn = np.sign(np.linalg.det(u @ vt))
    dmat = np.diag([1.0, 1.0, sgn])
    r = u @ dmat @ vt
    var_s = (xs**2).sum() / n
    s = float((d_diag * np.array([1.0, 1.0, sgn])).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * (r @ mu_s)
    aligned = (s * (r @ src.T)).T + t
    rmse_before = float(np.sqrt(((src - dst) ** 2).sum(1).mean()))
    rmse_after = float(np.sqrt(((aligned - dst) ** 2).sum(1).mean()))
    return {
        "R": r,
        "t": t,
        "s": s,
        "aligned": aligned,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "residual_ratio": rmse_after / rmse_before if rmse_before > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Frame matching by physical position
# --------------------------------------------------------------------------- #
def match_by_position(fwd_pos: np.ndarray, bwd_pos: np.ndarray, max_dist: float = 0.3):
    """Match each forward frame to its nearest backward frame by 3D position.

    Returns (pairs, dists) where pairs is a list of (i, j) and dists the
    matched euclidean distances. Only pairs within max_dist are kept.
    """
    fwd_pos = np.asarray(fwd_pos, dtype=np.float64)
    bwd_pos = np.asarray(bwd_pos, dtype=np.float64)
    d = np.linalg.norm(fwd_pos[:, None, :] - bwd_pos[None, :, :], axis=-1)  # (F, B)
    j = d.argmin(1)
    dmin = d[np.arange(len(fwd_pos)), j]
    pairs = [(int(i), int(j[i])) for i in range(len(fwd_pos)) if dmin[i] <= max_dist]
    dists = [float(dmin[i]) for i in range(len(fwd_pos)) if dmin[i] <= max_dist]
    return pairs, dists


# --------------------------------------------------------------------------- #
# Categorical posterior divergence (RSSM stochastic latent)
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def categorical_js(logit_a: np.ndarray, logit_b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two RSSM posteriors.

    logits shape (groups, classes) e.g. (32, 16). Returns mean JS over groups
    in nats, range [0, ln 2 ~= 0.693]. Symmetric, bounded — robust for
    comparing stochastic categorical states.
    """
    p = _softmax(np.asarray(logit_a, dtype=np.float64), -1)
    q = _softmax(np.asarray(logit_b, dtype=np.float64), -1)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum(-1)
    kl_qm = (q * (np.log(q + eps) - np.log(m + eps))).sum(-1)
    return float((0.5 * kl_pm + 0.5 * kl_qm).mean())


# --------------------------------------------------------------------------- #
# Camera pose -> world transform of WP (consistency check, optional)
# --------------------------------------------------------------------------- #
def quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def compare_pair(
    fwd_wp,
    bwd_wp,
    fwd_cp,
    bwd_cp,
    fwd_logit,
    bwd_logit,
    fwd_deter,
    bwd_deter,
    fwd_embed=None,
    bwd_embed=None,
):
    """Full metric bundle for one matched (forward, backward) frame pair."""
    fwp = np.asarray(fwd_wp).reshape(-1, 3)
    bwp = np.asarray(bwd_wp).reshape(-1, 3)
    align = umeyama(bwp, fwp, with_scale=True)
    out = {
        "wp_cosine": cosine(fwd_wp, bwd_wp),
        "wp_l2": l2(fwd_wp, bwd_wp),
        "cp_l2": l2(fwd_cp, bwd_cp),
        "cp_cosine": cosine(fwd_cp, bwd_cp),
        "wp_rmse_raw": align["rmse_before"],
        "wp_rmse_aligned": align["rmse_after"],
        "wp_residual_ratio": align["residual_ratio"],
        "latent_js": categorical_js(fwd_logit, bwd_logit),
        "deter_cosine": cosine(fwd_deter, bwd_deter),
        "deter_l2": l2(fwd_deter, bwd_deter),
    }
    if fwd_embed is not None and bwd_embed is not None:
        out["embed_cosine"] = cosine(fwd_embed, bwd_embed)
        out["embed_l2"] = l2(fwd_embed, bwd_embed)
    return out
