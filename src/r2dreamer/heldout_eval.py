"""Held-out world-model evaluation for the 3D-26 ablation.

Computes the metrics requested by issue 3D-26:

    rows    = {WP/CP, Aggregator}
    cols    = recon NLL, dynamics KL, representation KL, reward MSE,
              k-step latent rollout error for k ∈ {1, 5, 15}
    values  = mean ± std over 3 seeds; per-seed dumped to CSV

The training script's existing `agent.eval_loss(batch, key)` already produces
`loss/dyn` (dynamics KL) and `loss/rep` (representation KL); this module adds:

  * reward_mse — mean-squared error of the twohot reward head's expected
    value against the ground-truth rewards in the batch (the trainer logs
    the twohot loss; spec asks for raw MSE).
  * k_step_rollout_error[k] for k ∈ {1, 5, 15} — start from the posterior
    state at step t=0, run `rssm.img_step` with the ground-truth action
    sequence for k steps, then MSE on the RSSM feature against the
    ground-truth posterior at step k.
  * reconstruction NLL is **not** included — R2Dreamer in this repo has no
    pixel/embedding decoder, so there is no log-likelihood to evaluate.
    The aggregator script writes "N/A" for that cell.

Designed to be called once at end of training on a frozen agent.
"""

from __future__ import annotations

from typing import Any, Iterable

import jax
import jax.numpy as jnp

from src.r2dreamer.obs_batch import obs_leading_shape


def _reward_mse(agent: Any, batch: dict, rng_key: jnp.ndarray) -> float:
    """One batch -> scalar reward MSE between the head's mean and the GT reward.

    Uses `R2TwoHotDist.pred()` to recover the expected reward in real space
    from the twohot logits, then averages the squared error.
    """
    params = agent.params
    forward = agent._world_model_forward(params, batch, rng_key)
    B, T = obs_leading_shape(batch["obs"])
    feat = forward.feat
    rew_logits = (
        agent._modules["reward"]
        .apply(
            params["reward"],
            feat.reshape(B * T, -1),
        )
        .reshape(B, T, -1)
    )
    pred = agent.twohot.pred(rew_logits).reshape(B, T)
    err = pred - batch["rewards"]
    return float(jnp.mean(err * err))


def _k_step_rollout_error(
    agent: Any,
    batch: dict,
    rng_key: jnp.ndarray,
    k_values: tuple[int, ...] = (1, 5, 15),
) -> dict[int, float]:
    """For each k, MSE on RSSM features after rolling out k prior steps from t=0.

    The metric measures how well the prior dynamics + ground-truth actions
    predict the next-step latent without seeing observations after t=0.
    """
    params = agent.params
    forward = agent._world_model_forward(params, batch, rng_key)
    post_stochs = forward.post_stochs  # (B, T, C, K)
    post_deters = forward.post_deters  # (B, T, D)

    T = post_stochs.shape[1]
    max_k = max(k_values)
    if T < max_k + 1:
        raise ValueError(
            f"seq_len T={T} is too short for max k_rollout={max_k}; need T >= max_k+1"
        )

    actions = batch["actions"]  # (B, T, A) one-hot
    rssm = agent.rssm_mod
    rssm_params = params["rssm"]

    # Start from posterior at t=0; roll forward with prior dynamics.
    stoch = post_stochs[:, 0]
    deter = post_deters[:, 0]

    errors: dict[int, float] = {}
    for step in range(1, max_k + 1):
        stoch, deter = rssm.apply(
            rssm_params,
            stoch,
            deter,
            actions[:, step - 1],
            method=rssm.img_step,
            rngs={"sample": jax.random.fold_in(rng_key, step)},
        )
        if step in k_values:
            feat_pred = rssm.apply(rssm_params, stoch, deter, method=rssm.get_feat)
            feat_gt = rssm.apply(
                rssm_params,
                post_stochs[:, step],
                post_deters[:, step],
                method=rssm.get_feat,
            )
            err = jnp.mean(jnp.square(feat_pred - feat_gt))
            errors[step] = float(err)
    return errors


def compute_heldout_metrics(
    agent: Any,
    batches: Iterable[dict],
    *,
    rng_key: jnp.ndarray,
    k_values: tuple[int, ...] = (1, 5, 15),
) -> dict[str, float]:
    """Aggregate heldout metrics across batches.

    Each `batch` must already be `convert_batch`'d (obs/actions one-hot/
    rewards/is_first/is_last/is_terminal). Pulls eval_loss + reward_mse +
    k-step rollout per batch and averages.
    """
    eval_sums: dict[str, float] = {}
    rew_mse_sum = 0.0
    rollout_sums: dict[int, float] = {k: 0.0 for k in k_values}
    n = 0
    for batch in batches:
        rng_key, k_eval, k_rew, k_roll = jax.random.split(rng_key, 4)
        # Standard eval losses (dyn KL, rep KL, twohot reward loss, etc.).
        metrics = agent.eval_loss(batch, k_eval)
        for k, v in metrics.items():
            eval_sums[k] = eval_sums.get(k, 0.0) + float(v)
        rew_mse_sum += _reward_mse(agent, batch, k_rew)
        for k, v in _k_step_rollout_error(
            agent, batch, k_roll, k_values=k_values
        ).items():
            rollout_sums[k] += v
        n += 1

    if n == 0:
        return {}

    out: dict[str, float] = {f"heldout/{k}": v / n for k, v in eval_sums.items()}
    out["heldout/reward_mse"] = rew_mse_sum / n
    for k, total in rollout_sums.items():
        out[f"heldout/k_step_rollout_mse/k{k}"] = total / n
    out["heldout/reconstruction_nll"] = float(
        "nan"
    )  # No decoder — see module docstring.
    out["heldout/num_batches"] = float(n)
    return out


def metrics_table_row(metrics: dict[str, float]) -> dict[str, float]:
    """Translate the heldout output into the spec's table column names."""
    return {
        "reconstruction_nll": metrics.get("heldout/reconstruction_nll", float("nan")),
        "dynamics_kl": metrics.get("heldout/loss/dyn", float("nan")),
        "representation_kl": metrics.get("heldout/loss/rep", float("nan")),
        "reward_mse": metrics.get("heldout/reward_mse", float("nan")),
        "k_step_rollout_k1": metrics.get("heldout/k_step_rollout_mse/k1", float("nan")),
        "k_step_rollout_k5": metrics.get("heldout/k_step_rollout_mse/k5", float("nan")),
        "k_step_rollout_k15": metrics.get(
            "heldout/k_step_rollout_mse/k15", float("nan")
        ),
    }
