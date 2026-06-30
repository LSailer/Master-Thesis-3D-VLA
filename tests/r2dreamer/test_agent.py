"""Smoke tests for R2DreamerAgent."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.obs_batch import ObservationPacker, encoder_obs_from_batch
from src.r2dreamer.obs_batch import CAMERA_POSE_KEY, HOUSE_CONTEXT_KEY
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
        "is_episode_end": jnp.zeros((B, T)),
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
    is_episode_end = jnp.zeros((B, T), dtype=jnp.float32).at[:, -1].set(1.0)
    return {
        "obs": obs,
        "actions": jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        "rewards": rewards,
        "is_first": is_first,
        "is_episode_end": is_episode_end,
    }


def make_small_decoder_cfg(*, decoder=False):
    return R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=5,
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
        lr=1e-3,
        warmup_steps=0,
        decoder=decoder,
    )


def make_tiny_train_cfg(**overrides):
    params = {
        "num_actions": 4,
        "deter_size": 32,
        "hidden_size": 16,
        "stoch_classes": 4,
        "stoch_discrete": 4,
        "blocks": 4,
        "vggt_embed_dim": 8,
        "mlp_vggt_hidden": 8,
        "mlp_vggt_layers": 1,
        "mlp_units": 16,
        "mlp_layers_reward": 1,
        "mlp_layers_cont": 1,
        "mlp_layers_actor": 1,
        "mlp_layers_critic": 1,
        "twohot_bins": 21,
        "imagination_horizon": 2,
        "horizon": 10,
        "lr": 1e-3,
        "warmup_steps": 0,
    }
    params.update(overrides)
    return R2DreamerConfig(**params)


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
        -1.0, 1.0, num=B * T * VGGT_FEATURE_DIM, dtype=np.float32
    ).reshape((B, T, VGGT_FEATURE_DIM))
    action_ids = jnp.arange(B * T).reshape(B, T) % cfg.num_actions
    return {
        "obs": {
            "image": jnp.asarray(image),
            "wp_cp": jnp.asarray(wp_cp),
        },
        "actions": jax.nn.one_hot(action_ids, cfg.num_actions, dtype=jnp.float32),
        "rewards": jnp.zeros((B, T), dtype=jnp.float32),
        "is_first": jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0),
        "is_episode_end": jnp.zeros((B, T), dtype=jnp.float32),
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
        obs = {
            "image": np.random.randint(0, 256, cfg.obs_shape, dtype=np.uint8),
            "is_first": True,
        }
        encoder_obs = ObservationPacker(cfg).from_step(obs)
        action = agent.act(encoder_obs, obs["is_first"], jax.random.PRNGKey(0))
        assert 0 <= action < cfg.num_actions

    def test_act_with_state_matches_mutable_act_and_jits(self, agent, cfg):
        state = agent.initial_act_state()
        packer = ObservationPacker(cfg)
        image = np.random.randint(0, 256, cfg.obs_shape, dtype=np.uint8)
        for step, is_first in enumerate((True, False, True, False)):
            obs = {"image": image, "is_first": is_first}
            encoder_obs = packer.from_step(obs)
            key = jax.random.PRNGKey(step)
            mutable_action = agent.act(encoder_obs, is_first, key, training=False)
            state_action, state = agent.act_with_state(
                encoder_obs, is_first, state, key, training=False
            )
            assert state_action == mutable_action
            assert tree_allclose(state, agent.snapshot_act_state())

        obs = {"image": image, "is_first": True}
        compiled = jax.jit(agent.act_with_state_pure)
        action, _state = compiled.__call__(
            agent.params,
            packer.from_step(obs),
            agent.initial_act_state(),
            jnp.asarray(True),
            jax.random.PRNGKey(99),
            False,
        )
        assert 0 <= int(action) < cfg.num_actions

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
            obs_shape=(13312,),
            num_actions=4,
            deter_size=32,
            hidden_size=16,
            stoch_classes=4,
            stoch_discrete=4,
            blocks=4,
            encoder_depth=8,
            encoder_kernel=3,
            encoder_mults=(1, 1),
            vggt_feature_dim=1024,
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
                "house_context": jnp.zeros((1, 2, 1024), dtype=jnp.float32),
            },
            "actions": jax.nn.one_hot(
                jnp.zeros((1, 2), dtype=jnp.int32), cfg.num_actions
            ),
            "rewards": jnp.zeros((1, 2), dtype=jnp.float32),
            "is_first": jnp.ones((1, 2), dtype=jnp.float32),
            "is_episode_end": jnp.zeros((1, 2), dtype=jnp.float32),
        }

        metrics = agent.train_step(batch, jax.random.PRNGKey(1))

        assert np.isfinite(metrics["total_loss"])
        assert "hybrid/vggt_frac" in metrics

    def test_train_step_accepts_static_house_points_with_camera_pose(self):
        cfg = make_tiny_train_cfg(
            encoder_type="vggt_house_points_pose",
            obs_shape={CAMERA_POSE_KEY: (9,), HOUSE_CONTEXT_KEY: (5, 6)},
        )
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
        batch = {
            "obs": {
                CAMERA_POSE_KEY: jnp.zeros((1, 2, 9), dtype=jnp.float16),
                HOUSE_CONTEXT_KEY: jnp.ones((5, 6), dtype=jnp.float16),
            },
            "actions": jax.nn.one_hot(
                jnp.zeros((1, 2), dtype=jnp.int32), cfg.num_actions
            ),
            "rewards": jnp.zeros((1, 2), dtype=jnp.float32),
            "is_first": jnp.ones((1, 2), dtype=jnp.float32),
            "is_episode_end": jnp.zeros((1, 2), dtype=jnp.float32),
        }

        encoder_obs = encoder_obs_from_batch(batch, cfg)
        assert encoder_obs[CAMERA_POSE_KEY].shape == (2, 9)
        assert encoder_obs[HOUSE_CONTEXT_KEY].shape == (1, 5, 6)

        metrics = agent.train_step(batch, jax.random.PRNGKey(1))

        assert np.isfinite(metrics["total_loss"])

    @pytest.mark.parametrize(
        "encoder_type, token_key, token_shape",
        [
            ("vggt_house_full_tokens_nogate", "full_tokens", (1, 2, 6, 8)),
            ("vggt_house_global_tokens_nogate", "global_tokens", (1, 6, 8)),
        ],
    )
    def test_train_step_accepts_live_tokens_without_gate(
        self, encoder_type, token_key, token_shape
    ):
        cfg = R2DreamerConfig(
            encoder_type=encoder_type,
            obs_shape={"image": (3, 64, 64), token_key: (6, 8)},
            num_actions=4,
            deter_size=32,
            hidden_size=16,
            stoch_classes=4,
            stoch_discrete=4,
            blocks=4,
            encoder_depth=2,
            encoder_kernel=3,
            encoder_mults=(1, 1),
            vggt_embed_dim=8,
            vggt_token_count=6,
            vggt_token_dim=8,
            vggt_token_transformer_layers=1,
            vggt_token_transformer_heads=2,
            vggt_token_transformer_mlp_ratio=2,
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
                token_key: jnp.zeros(token_shape, dtype=jnp.float32),
            },
            "actions": jax.nn.one_hot(
                jnp.zeros((1, 2), dtype=jnp.int32), cfg.num_actions
            ),
            "rewards": jnp.zeros((1, 2), dtype=jnp.float32),
            "is_first": jnp.ones((1, 2), dtype=jnp.float32),
            "is_episode_end": jnp.zeros((1, 2), dtype=jnp.float32),
        }

        before = agent.params["encoder"]
        metrics = agent.train_step(batch, jax.random.PRNGKey(1))
        after = agent.params["encoder"]

        assert np.isfinite(metrics["total_loss"])
        assert "hybrid/gate" not in metrics
        assert "gate" not in after["params"]
        assert "token_transformer" in after["params"]
        assert not jax.tree_util.tree_all(
            jax.tree.map(
                lambda a, b: jnp.allclose(a, b),
                before["params"]["token_transformer"],
                after["params"]["token_transformer"],
            )
        )

    def test_full_token_nogate_uses_configured_bfloat16_compute(self):
        cfg = R2DreamerConfig(
            encoder_type="vggt_house_full_tokens_nogate",
            obs_shape={"image": (3, 64, 64), "full_tokens": (6, 8)},
            num_actions=4,
            encoder_depth=2,
            encoder_kernel=3,
            encoder_mults=(1, 1),
            vggt_embed_dim=8,
            vggt_token_count=6,
            vggt_token_dim=8,
            vggt_token_transformer_layers=1,
            vggt_token_transformer_heads=2,
            vggt_token_transformer_mlp_ratio=2,
            compute_dtype="bfloat16",
        )
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
        batch = {
            "obs": {
                "image": jnp.zeros((1, 2, 3, 64, 64), dtype=jnp.uint8),
                "full_tokens": jnp.zeros((1, 2, 6, 8), dtype=jnp.float32),
            }
        }

        encoder_obs = encoder_obs_from_batch(batch, cfg)
        _, token_e = agent.encoder_mod.apply(
            {"params": agent.params["encoder"]["params"]},
            encoder_obs,
            method=agent.encoder_mod.branches,
        )

        assert encoder_obs["full_tokens"].dtype == jnp.bfloat16
        assert token_e.dtype == jnp.bfloat16

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

    def test_decoder_probe_updates_only_decoder_params(self):
        cfg = make_small_decoder_cfg(decoder=False)
        dec_cfg = make_small_decoder_cfg(decoder=True)
        batch = make_deterministic_batch(cfg, B=1, T=2)
        init_rng = jax.random.PRNGKey(17)
        train_rng = jax.random.PRNGKey(19)

        baseline = R2DreamerAgent(cfg, init_rng)
        with_decoder = R2DreamerAgent(dec_cfg, init_rng)
        before_decoder_params = jax.tree.map(jnp.copy, with_decoder.params["decoder"])

        baseline_metrics = baseline.train_step(batch, train_rng)
        decoder_metrics = with_decoder.train_step(batch, train_rng)

        assert "loss/decoder" in decoder_metrics
        assert np.isfinite(decoder_metrics["loss/decoder"])
        expected_agent_loss = (
            dec_cfg.scale_dyn * decoder_metrics["loss/dyn"]
            + dec_cfg.scale_rep * decoder_metrics["loss/rep"]
            + dec_cfg.scale_barlow * decoder_metrics["loss/barlow"]
            + dec_cfg.scale_rew * decoder_metrics["loss/rew"]
            + dec_cfg.scale_con * decoder_metrics["loss/con"]
            + dec_cfg.scale_policy * decoder_metrics["loss/policy"]
            + dec_cfg.scale_value * decoder_metrics["loss/value"]
            + dec_cfg.scale_repval * decoder_metrics["loss/repval"]
        )
        assert decoder_metrics["total_loss"] == pytest.approx(expected_agent_loss, rel=1e-6)
        assert decoder_metrics["opt_loss"] == pytest.approx(
            expected_agent_loss + dec_cfg.scale_decoder * decoder_metrics["loss/decoder"],
            rel=1e-6,
        )
        assert tree_any_changed(before_decoder_params, with_decoder.params["decoder"])
        for name in baseline.params:
            assert tree_allclose(
                baseline.params[name], with_decoder.params[name], atol=1e-6
            ), f"decoder probe changed non-decoder subtree {name}"
        assert baseline_metrics["nan_skipped"] == decoder_metrics["nan_skipped"] == 0.0
