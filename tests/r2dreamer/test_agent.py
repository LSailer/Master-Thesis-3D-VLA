"""Smoke tests for R2DreamerAgent."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.adapters.hybrid_adapter import HYBRID_FEATURE_DIM
from src.r2dreamer.adapters.vggt_adapter import VGGT_FEATURE_DIM


@pytest.fixture
def cfg():
    return R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=17)


@pytest.fixture
def agent(cfg):
    return R2DreamerAgent(cfg, jax.random.PRNGKey(42))


def make_batch(cfg, B=4, T=16):
    return {
        "obs": jnp.array(np.random.rand(B, T, *cfg.obs_shape).astype(np.float32)),
        "actions": jnp.array(np.eye(cfg.num_actions, dtype=np.float32)[
            np.random.randint(0, cfg.num_actions, (B, T))]),
        "rewards": jnp.array(np.random.randn(B, T).astype(np.float32)),
        "is_first": jnp.zeros((B, T)),
        "is_last": jnp.zeros((B, T)),
        "is_terminal": jnp.zeros((B, T)),
    }


def make_deterministic_cfg():
    return R2DreamerConfig(
        obs_shape=(3, 16, 16),
        num_actions=5,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=8,
        encoder_kernel=3,
        encoder_mults=(1, 1),
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=3,
        horizon=20,
        lr=1e-3,
        warmup_steps=0,
    )


def make_deterministic_batch(cfg, B=2, T=4):
    obs_size = B * T * np.prod(cfg.obs_shape)
    obs = jnp.linspace(0.0, 1.0, obs_size, dtype=jnp.float32).reshape(
        B, T, *cfg.obs_shape
    )
    action_ids = jnp.arange(B * T).reshape(B, T) % cfg.num_actions
    rewards = jnp.linspace(-1.0, 1.0, B * T, dtype=jnp.float32).reshape(B, T)
    is_first = jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0)
    is_last = jnp.zeros((B, T), dtype=jnp.float32).at[:, -1].set(1.0)
    is_terminal = jnp.zeros((B, T), dtype=jnp.float32).at[0, -1].set(1.0)
    return {
        "obs": obs,
        "actions": jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        "rewards": rewards,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }


def make_small_hybrid_cfg():
    return R2DreamerConfig(
        encoder_type="hybrid",
        obs_shape=(HYBRID_FEATURE_DIM,),
        num_actions=4,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1),
        vggt_embed_dim=16,
        mlp_vggt_hidden=16,
        mlp_vggt_layers=1,
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=3,
        horizon=20,
        lr=1e-3,
        warmup_steps=0,
    )


def make_hybrid_mapping_batch(cfg, B=1, T=4):
    image_values = np.arange(B * T * 3 * 64 * 64, dtype=np.uint32) % 256
    image = image_values.astype(np.uint8).reshape(B, T, 3, 64, 64)
    wp_cp = np.linspace(
        -1.0, 1.0, B * T * VGGT_FEATURE_DIM, dtype=np.float32
    ).reshape(B, T, VGGT_FEATURE_DIM)
    action_ids = jnp.arange(B * T).reshape(B, T) % cfg.num_actions
    return {
        "obs": {
            "image": jnp.asarray(image),
            "wp_cp": jnp.asarray(wp_cp),
        },
        "actions": jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        "rewards": jnp.zeros((B, T), dtype=jnp.float32),
        "is_first": jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0),
        "is_last": jnp.zeros((B, T), dtype=jnp.float32),
        "is_terminal": jnp.zeros((B, T), dtype=jnp.float32),
    }


def tree_allclose(left, right, *, atol=1e-6):
    pairs = zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    return all(np.allclose(np.asarray(a), np.asarray(b), atol=atol) for a, b in pairs)


def tree_any_changed(before, after, *, atol=1e-7):
    pairs = zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after))
    return any(not np.allclose(np.asarray(a), np.asarray(b), atol=atol) for a, b in pairs)


class TestR2DreamerAgent:
    def test_init(self, agent):
        assert agent is not None

    def test_act(self, agent, cfg):
        obs = {"image": np.random.randint(0, 256, cfg.obs_shape, dtype=np.uint8), "is_first": True}
        action = agent.act(obs, jax.random.PRNGKey(0))
        assert 0 <= action < cfg.num_actions

    def test_train_step_produces_metrics(self, agent, cfg):
        batch = make_batch(cfg)
        metrics = agent.train_step(batch, jax.random.PRNGKey(1))
        assert "loss/barlow" in metrics
        assert "loss/dyn" in metrics
        assert "loss/rew" in metrics
        assert "loss/policy" in metrics
        assert "loss/value" in metrics
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} = {v}"

    def test_train_step_does_not_diverge(self, agent, cfg):
        batch = make_batch(cfg)
        rng = jax.random.PRNGKey(2)
        losses = []
        for _ in range(3):
            rng, k = jax.random.split(rng)
            m = agent.train_step(batch, k)
            losses.append(m["total_loss"])
        # Should not be NaN or explode
        assert all(np.isfinite(l) for l in losses)

    def test_train_step_accepts_vggt_house_context_batch(self):
        cfg = R2DreamerConfig(
            encoder_type="vggt_house_context",
            obs_shape=(16404,),
            num_actions=4,
            deter_size=32,
            hidden_size=16,
            stoch_classes=4,
            stoch_discrete=4,
            blocks=4,
            encoder_depth=8,
            encoder_kernel=3,
            encoder_mults=(1, 1),
            vggt_embed_dim=8,
            mlp_vggt_hidden=8,
            mlp_vggt_layers=1,
            mlp_units=16,
            mlp_layers_reward=1,
            mlp_layers_cont=1,
            mlp_layers_actor=1,
            mlp_layers_critic=1,
            twohot_bins=21,
            imagination_horizon=2,
            horizon=10,
            lr=1e-3,
            warmup_steps=0,
        )
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
        batch = {
            "obs": {
                "image": jnp.zeros((1, 2, 3, 64, 64), dtype=jnp.float32),
                "house_context": jnp.zeros((1, 2, 4116), dtype=jnp.float32),
            },
            "actions": jax.nn.one_hot(
                jnp.zeros((1, 2), dtype=jnp.int32), cfg.num_actions
            ),
            "rewards": jnp.zeros((1, 2), dtype=jnp.float32),
            "is_first": jnp.ones((1, 2), dtype=jnp.float32),
            "is_last": jnp.zeros((1, 2), dtype=jnp.float32),
            "is_terminal": jnp.zeros((1, 2), dtype=jnp.float32),
        }

        metrics = agent.train_step(batch, jax.random.PRNGKey(1))

        assert np.isfinite(metrics["total_loss"])
        assert "hybrid/vggt_frac" in metrics

    def test_train_step_is_deterministic_updates_params_and_composes_total_loss(self):
        cfg = make_deterministic_cfg()
        batch = make_deterministic_batch(cfg)
        init_rng = jax.random.PRNGKey(7)
        train_rng = jax.random.PRNGKey(11)
        agent_a = R2DreamerAgent(cfg, init_rng)
        agent_b = R2DreamerAgent(cfg, init_rng)
        before = jax.tree.map(jnp.copy, agent_a.params)

        metrics_a = agent_a.train_step(batch, train_rng)
        metrics_b = agent_b.train_step(batch, train_rng)

        assert metrics_a.keys() == metrics_b.keys()
        for key in metrics_a:
            assert metrics_a[key] == pytest.approx(metrics_b[key], abs=1e-6)
        assert tree_allclose(agent_a.params, agent_b.params)

        assert metrics_a["nan_skipped"] == 0.0
        changed_subtrees = [
            name for name, params in agent_a.params.items()
            if tree_any_changed(before[name], params)
        ]
        assert changed_subtrees

        expected_total = (
            cfg.scale_dyn * metrics_a["loss/dyn"]
            + cfg.scale_rep * metrics_a["loss/rep"]
            + cfg.scale_barlow * metrics_a["loss/barlow"]
            + cfg.scale_rew * metrics_a["loss/rew"]
            + cfg.scale_con * metrics_a["loss/con"]
            + cfg.scale_policy * metrics_a["loss/policy"]
            + cfg.scale_value * metrics_a["loss/value"]
            + cfg.scale_repval * metrics_a["loss/repval"]
        )
        assert metrics_a["total_loss"] == pytest.approx(expected_total, rel=1e-6)

    def test_hybrid_train_step_accepts_mapping_obs(self):
        cfg = make_small_hybrid_cfg()
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(5))
        batch = make_hybrid_mapping_batch(cfg)

        metrics = agent.train_step(batch, jax.random.PRNGKey(6))

        assert metrics["nan_skipped"] == 0.0
        assert "hybrid/gate" in metrics
        assert "hybrid/cnn_frac" in metrics
        assert "hybrid/vggt_frac" in metrics
        assert np.isfinite(metrics["total_loss"])
