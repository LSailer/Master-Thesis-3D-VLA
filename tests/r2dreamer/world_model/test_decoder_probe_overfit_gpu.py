"""GPU overfit probe for the stop-gradient RGB decoder.

This is intentionally marked GPU and excluded from the normal CPU suite. It
checks that ``--decoder`` can learn to visualise a fixed latent without relying
on the agent losses: all non-decoder loss scales are zero, so the only useful
update is the decoder-only reconstruction objective.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from src.buffer.replay_buffer import ReplayBatch
from src.r2dreamer.agent import R2DreamerAgent
from src.configs.config import R2DreamerConfig


pytestmark = pytest.mark.gpu


def _decoder_probe_cfg() -> R2DreamerConfig:
    return R2DreamerConfig(
        obs_shape=(64, 64, 3),
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


def _structured_rgb_batch(
    cfg: R2DreamerConfig, *, B: int = 2, T: int = 3
) -> ReplayBatch:
    y = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, :, None]
    x = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, None, :]
    base = jnp.stack([
        jnp.broadcast_to(x, (1, 64, 64))[0],
        jnp.broadcast_to(y, (1, 64, 64))[0],
        jnp.broadcast_to((x + y) * 0.5, (1, 64, 64))[0],
    ], axis=-1)  # (64, 64, 3) HWC
    frames = []
    for i in range(B * T):
        frames.append(jnp.roll(base, shift=i * 3, axis=1))
    obs = jnp.stack(frames, axis=0).reshape(B, T, 64, 64, 3)
    action_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % cfg.num_actions
    return ReplayBatch(
        obs=obs,
        actions=jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        rewards=jnp.zeros((B, T), dtype=jnp.float32),
        is_first=jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0),
        is_episode_end=jnp.zeros((B, T), dtype=jnp.float32).at[:, -1].set(1.0),
    )


def _habitat_rgb_batch(
    cfg: R2DreamerConfig, *, B: int = 1, T: int = 8
) -> ReplayBatch:
    fixture = (
        Path(__file__).parents[1]
        / "launch"
        / "fixtures"
        / "sample_habitat_obs.npz"
    )
    # The npz fixture predates the HWC flip and stores (10, 3, 518, 518) CHW.
    frames = np.load(fixture)["frames"][: B * T]
    resized = []
    for frame in frames:
        hwc = np.transpose(frame, (1, 2, 0))
        resized_hwc = Image.fromarray(hwc).resize(
            (64, 64), Image.Resampling.BILINEAR
        )
        resized.append(np.asarray(resized_hwc))  # keep HWC
    obs = jnp.asarray(np.stack(resized).reshape(B, T, 64, 64, 3), dtype=jnp.uint8)
    action_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % cfg.num_actions
    return ReplayBatch(
        obs=obs,
        actions=jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        rewards=jnp.zeros((B, T), dtype=jnp.float32),
        is_first=jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0),
        is_episode_end=jnp.zeros((B, T), dtype=jnp.float32).at[:, -1].set(1.0),
    )


def _recon_mse(agent: R2DreamerAgent, batch: ReplayBatch) -> float:
    target, recon = agent.reconstruct(batch)
    return float(np.mean((recon - target) ** 2))


def _save_recon_grid(agent: R2DreamerAgent, batch: ReplayBatch, path: Path) -> None:
    target, recon = agent.reconstruct(batch)
    rows = []
    for i in range(target.shape[0]):
        # target/recon are (B*T, 64, 64, 3) HWC — PIL-ready without transposes.
        rows.append(np.concatenate([target[i], recon[i]], axis=1))
    grid = np.concatenate(rows, axis=0)
    image = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _run_decoder_overfit_probe(
    *,
    batch: ReplayBatch,
    steps: int,
    max_final_fraction: float,
    artifact_prefix: Path | None = None,
) -> tuple[float, float]:
    cfg = _decoder_probe_cfg()
    agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))

    initial = _recon_mse(agent, batch)
    if artifact_prefix is not None:
        _save_recon_grid(
            agent,
            batch,
            artifact_prefix.with_name(
                f"{artifact_prefix.name}_step_000_input_left_recon_right.png"
            ),
        )

    rng = jax.random.PRNGKey(1)
    metrics = {}
    for _ in range(steps):
        rng, step_key = jax.random.split(rng)
        metrics = agent.train_step(batch, step_key)
    final = _recon_mse(agent, batch)

    if artifact_prefix is not None:
        final_path = artifact_prefix.with_name(
            f"{artifact_prefix.name}_step_{steps:03d}_input_left_recon_right.png"
        )
        _save_recon_grid(agent, batch, final_path)
        assert final_path.exists()

    assert metrics["total_loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["opt_loss"] == pytest.approx(metrics["loss/decoder"], rel=1e-6)
    assert final < initial * max_final_fraction, (initial, final)
    return initial, final


def test_decoder_probe_overfits_fixed_latents_on_gpu():
    cfg = _decoder_probe_cfg()
    batch = _structured_rgb_batch(cfg)

    _run_decoder_overfit_probe(
        batch=batch,
        steps=120,
        max_final_fraction=0.75,
    )


def test_decoder_probe_saves_qualitative_habitat_reconstructions_on_gpu():
    cfg = _decoder_probe_cfg()
    batch = _habitat_rgb_batch(cfg, B=1, T=8)

    _run_decoder_overfit_probe(
        batch=batch,
        steps=5000,
        max_final_fraction=1.0,
        artifact_prefix=Path("artifacts/decoder_probe/habitat_decoder_overfit"),
    )
