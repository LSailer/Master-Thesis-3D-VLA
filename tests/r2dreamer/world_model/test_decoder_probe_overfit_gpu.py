"""GPU overfit probe for the stop-gradient RGB decoder.

This is intentionally marked GPU and excluded from the normal CPU suite. It
checks that ``--decoder`` can learn to visualise a fixed latent without relying
on the agent losses: all non-decoder loss scales are zero, so the only useful
update is the decoder-only reconstruction objective.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.config import R2DreamerConfig


pytestmark = pytest.mark.gpu


def _decoder_probe_cfg() -> R2DreamerConfig:
    return R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=4,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1, 1, 1),
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=2,
        horizon=10,
        lr=3e-3,
        warmup_steps=0,
        decoder=True,
        scale_decoder=1.0,
        # Freeze the agent objective. The decoder loss still enters opt_loss,
        # but its input feature is stop-gradient, so only decoder params move.
        scale_dyn=0.0,
        scale_rep=0.0,
        scale_barlow=0.0,
        scale_rew=0.0,
        scale_con=0.0,
        scale_policy=0.0,
        scale_value=0.0,
        scale_repval=0.0,
    )


def _structured_rgb_batch(cfg: R2DreamerConfig, *, B: int = 2, T: int = 3) -> dict:
    y = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, :, None]
    x = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, None, :]
    base = jnp.concatenate([
        jnp.broadcast_to(x, (1, 64, 64)),
        jnp.broadcast_to(y, (1, 64, 64)),
        jnp.broadcast_to((x + y) * 0.5, (1, 64, 64)),
    ], axis=0)
    frames = []
    for i in range(B * T):
        frames.append(jnp.roll(base, shift=i * 3, axis=2))
    obs = jnp.stack(frames, axis=0).reshape(B, T, 3, 64, 64)
    action_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % cfg.num_actions
    return {
        "obs": obs,
        "actions": jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        "rewards": jnp.zeros((B, T), dtype=jnp.float32),
        "is_first": jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0),
        "is_last": jnp.zeros((B, T), dtype=jnp.float32).at[:, -1].set(1.0),
        "is_terminal": jnp.zeros((B, T), dtype=jnp.float32),
    }


def _recon_mse(agent: R2DreamerAgent, batch: dict) -> float:
    target, recon = agent.reconstruct(batch)
    return float(np.mean((recon - target) ** 2))


def test_decoder_probe_overfits_fixed_latents_on_gpu():
    cfg = _decoder_probe_cfg()
    agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
    batch = _structured_rgb_batch(cfg)

    initial = _recon_mse(agent, batch)
    rng = jax.random.PRNGKey(1)
    metrics = {}
    for _ in range(120):
        rng, step_key = jax.random.split(rng)
        metrics = agent.train_step(batch, step_key)
    final = _recon_mse(agent, batch)

    assert metrics["total_loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["opt_loss"] == pytest.approx(metrics["loss/decoder"], rel=1e-6)
    assert final < initial * 0.75, (initial, final)
