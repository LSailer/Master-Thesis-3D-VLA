
"""Shape and integration tests for VGGT encoder + replay buffer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.buffer.replay_buffer import ReplayBatch, ReplayBuffer, ReplayTransition
from src.configs.config import R2DreamerConfig
from src.environments.observation import ObservationFrame
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.mlp import (
    MLPEncoder,
    VGGTAggregatorMLPEncoder,
    WP64CNNCPMLPEncoder,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import CAMERA_POSE_KEY, WORLD_POINTS_KEY
from src.r2dreamer.observation_preparation.vggt_readouts import full_aggregator_tokens

FEATURE_DIM = 4116  # 37*37*3 + 9
POOL_DIM = 1024
POOLED_FEATURE_DIM = 3 * POOL_DIM  # [cam | mean_patches | max_patches]


class _FullTokenOutput:
    """Minimal VGGT-output object for full-token readout tests."""

    world_points = None
    camera_pose = None

    def __init__(self, frame_tokens: jax.Array, global_tokens: jax.Array) -> None:
        self.frame_tokens = frame_tokens
        self.global_tokens = global_tokens


class TestTokenTransformerEncoder:
    """Shape tests for the generic token Transformer encoder."""

    def test_output_shape_with_reduced_token_count(self):
        enc = TokenTransformerEncoder(
            embed_dim=64,
            token_dim=16,
            num_tokens=10,
            model_dim=32,
            layers=2,
            heads=4,
            mlp_ratio=2,
            readout="camera_register_patch",
            norm_kind="rms",
            activation="silu",
            keep_register_tokens=True,
        )
        dummy = jnp.zeros((2, 10 * 16), dtype=jnp.float16)
        params = enc.init(jax.random.PRNGKey(0), dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 64)
        assert jnp.isfinite(out).all()

    def test_can_drop_register_tokens_for_future_ablation(self):
        enc = TokenTransformerEncoder(
            embed_dim=32,
            token_dim=8,
            num_tokens=10,
            model_dim=16,
            layers=1,
            heads=4,
            readout="camera_patch",
            norm_kind="rms",
            activation="silu",
            keep_register_tokens=False,
        )
        dummy = jnp.ones((1, 80), dtype=jnp.float16)
        params = enc.init(jax.random.PRNGKey(1), dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (1, 32)
        assert params["params"]["pos_embed"].shape == (1, 6, 16)

    def test_rejects_wrong_flat_dim(self):
        enc = TokenTransformerEncoder(
            embed_dim=32,
            token_dim=8,
            num_tokens=10,
            model_dim=16,
            heads=4,
        )
        with pytest.raises(ValueError, match="expected tokens"):
            enc.init(jax.random.PRNGKey(0), jnp.zeros((1, 79), dtype=jnp.float16))


class TestFullTokenContextTransformer:
    """Shape tests for a token-only context Transformer."""

    def test_full_token_source_shape(self):
        out = full_aggregator_tokens(
            _FullTokenOutput(
                frame_tokens=jnp.zeros((1374, 1024), dtype=jnp.float32),
                global_tokens=jnp.zeros((1374, 1024), dtype=jnp.float32),
            ),
            expected_shape=(1374, 2048),
        )
        assert out.shape == (1374, 2048)

    def test_output_shape_with_reduced_dims_and_no_token_projection(self):
        enc = TokenTransformerEncoder(
            embed_dim=32,
            token_dim=16,
            num_tokens=10,
            model_dim=None,
            layers=2,
            heads=4,
            mlp_ratio=2,
            dropout=0.0,
            readout="mean",
            norm_kind="layer",
            activation="gelu",
        )
        dummy = jnp.zeros((2, 10, 16), dtype=jnp.float32)
        params = enc.init(jax.random.PRNGKey(0), dummy, train=False)
        out = enc.apply(params, dummy, train=False)

        assert out.shape == (2, 32)
        assert "token_proj" not in params["params"]
        assert params["params"]["proj"]["kernel"].shape == (16, 32)

    def test_rejects_wrong_full_token_shape(self):
        enc = TokenTransformerEncoder(
            embed_dim=32,
            token_dim=16,
            num_tokens=10,
            heads=4,
        )
        with pytest.raises(ValueError, match="expected tokens"):
            enc.init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 15), dtype=jnp.float32))

    def test_bfloat16_compute_keeps_transformer_activations_bfloat16(self):
        enc = TokenTransformerEncoder(
            embed_dim=32,
            token_dim=16,
            num_tokens=10,
            layers=1,
            heads=4,
            mlp_ratio=2,
            dropout=0.0,
            compute_dtype=jnp.bfloat16,
        )
        dummy = jnp.zeros((2, 10, 16), dtype=jnp.float32)
        params = enc.init(jax.random.PRNGKey(0), dummy, train=False)
        out = enc.apply(params, dummy, train=False)

        assert out.dtype == jnp.bfloat16


class TestRGBTokenTransformerEncoder:
    """Shape tests for live RGB+token context fusion without a gate."""

    def test_accepts_image_and_live_full_tokens_without_gate(self):
        enc = TokenTransformerEncoder(
            embed_dim=32,
            token_dim=16,
            num_tokens=10,
            model_dim=None,
            layers=1,
            heads=4,
            mlp_ratio=2,
            token_key="full_tokens",
            image_key="image",
            cnn_depth=2,
            cnn_kernel=3,
            cnn_mults=(1, 1, 1, 1),
        )
        obs = {
            "image": jnp.zeros((2, 64, 64, 3), dtype=jnp.float32),
            "full_tokens": jnp.zeros((2, 10, 16), dtype=jnp.float32),
        }
        params = enc.init(jax.random.PRNGKey(0), obs)

        fused = enc.apply(params, obs)
        cnn_e, token_e = enc.apply(params, obs, method=enc.branches)

        assert cnn_e.shape == (2, 32)
        assert token_e.shape == (2, 32)
        assert fused.shape == (2, 64)
        assert "gate" not in params["params"]


class TestVGGTAggregatorMLPEncoder:
    """Test Variant 1 VGGT aggregator encoder over adapter-pooled features."""

    def test_output_shape(self):
        enc = VGGTAggregatorMLPEncoder(embed_dim=1024)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, POOLED_FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 1024)
        assert jnp.isfinite(out).all()

    def test_custom_embed_dim(self):
        enc = VGGTAggregatorMLPEncoder(embed_dim=512)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, POOLED_FEATURE_DIM), dtype=jnp.float32)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (1, 512)

    def test_rejects_unpooled_tokens(self):
        """The encoder no longer accepts (B, N, D) all-token input."""
        enc = VGGTAggregatorMLPEncoder(embed_dim=32)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, 1374, 1024), dtype=jnp.float32)
        with pytest.raises(ValueError, match="pooled features"):
            enc.init(rng, dummy)

    def test_per_pool_norms_and_mlp_params(self):
        # Default depth = 1 hidden block (hidden0/norm0) + linear readout (proj).
        enc = VGGTAggregatorMLPEncoder(embed_dim=32, pool_dim=8, hidden=16)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.arange(2 * 24, dtype=jnp.float32).reshape(2, 24)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)

        assert out.shape == (2, 32)
        assert set(params["params"].keys()) == {
            "norm_cam",
            "norm_mean",
            "norm_max",
            "hidden0",
            "norm0",
            "proj",
        }

    def test_num_layers_stacks_blocks(self):
        # num_layers=3 -> hidden0/1/2 + norm0/1/2 + per-pool norms + proj.
        enc = VGGTAggregatorMLPEncoder(
            embed_dim=32, pool_dim=8, hidden=16, num_layers=3
        )
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, 24), dtype=jnp.float32)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 32)
        assert set(params["params"].keys()) == {
            "norm_cam",
            "norm_mean",
            "norm_max",
            "hidden0",
            "hidden1",
            "hidden2",
            "norm0",
            "norm1",
            "norm2",
            "proj",
        }


class TestMLPEncoder:
    """Test generic flat-feature MLP encoder."""

    def test_output_shape(self):
        enc = MLPEncoder(embed_dim=1024)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 1024)

    def test_batched(self):
        enc = MLPEncoder(embed_dim=1024)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((8, FEATURE_DIM)) * 0.5
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (8, 1024)
        assert jnp.isfinite(out).all()

    def test_custom_embed_dim(self):
        enc = MLPEncoder(embed_dim=512)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((1, FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (1, 512)

    def test_default_is_one_hidden_block_plus_proj(self):
        # 3D-52: default depth replaces the bare linear projection with an MLP.
        enc = MLPEncoder(embed_dim=64, hidden=48)
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros((1, FEATURE_DIM)))
        assert set(params["params"].keys()) == {"hidden0", "norm0", "proj"}

    def test_num_layers_three(self):
        enc = MLPEncoder(embed_dim=64, hidden=48, num_layers=3)
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros((2, FEATURE_DIM)))
        out = enc.apply(params, jnp.zeros((2, FEATURE_DIM)))
        assert out.shape == (2, 64)
        assert set(params["params"].keys()) == {
            "hidden0",
            "hidden1",
            "hidden2",
            "norm0",
            "norm1",
            "norm2",
            "proj",
        }

    def test_num_layers_zero_is_bare_linear(self):
        # Escape hatch: depth 0 reproduces the historical single-Dense projection.
        enc = MLPEncoder(embed_dim=64, num_layers=0)
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros((1, FEATURE_DIM)))
        assert set(params["params"].keys()) == {"proj"}


class TestWP64CNNCPMLPObservationBatch:
    def test_encoder_preserves_replay_leading_dims_for_structured_fields(self):
        enc = WP64CNNCPMLPEncoder(
            embed_dim=16,
            conv_depth=2,
            conv_kernel=3,
            conv_mults=(1, 1),
            cp_hidden=8,
            cp_layers=1,
        )
        obs = {
            WORLD_POINTS_KEY: jnp.ones((2, 3, 64, 64, 3), dtype=jnp.float16),
            CAMERA_POSE_KEY: jnp.ones((2, 3, 9), dtype=jnp.float16),
        }

        params = enc.init(jax.random.PRNGKey(0), obs)
        out = enc.apply(params, obs)

        assert out.shape == (2, 3, 16)
        assert jnp.isfinite(out).all()


class TestVGGTObservationBatchDType:
    def test_flat_vggt_encoder_flattens_structured_wp_cp_internally(self):
        enc = MLPEncoder(embed_dim=16, hidden=16, num_layers=1)
        obs = {
            WORLD_POINTS_KEY: jnp.ones((2, 3, 3, 37, 37), dtype=jnp.float16),
            CAMERA_POSE_KEY: jnp.ones((2, 3, 9), dtype=jnp.float16),
        }

        params = enc.init(jax.random.PRNGKey(0), obs)
        out = enc.apply(params, obs)

        assert out.shape == (2, 3, 16)
        assert jnp.isfinite(out).all()


class TestWP64CNNCPMLPEncoder:
    def test_world_points_and_camera_pose_fuse_to_embed(self):
        enc = WP64CNNCPMLPEncoder(
            embed_dim=64,
            conv_depth=4,
            conv_mults=(2, 2, 2, 2),
            cp_hidden=32,
            cp_layers=1,
        )
        obs = {
            WORLD_POINTS_KEY: jnp.ones((2, 64, 64, 3), dtype=jnp.float32),
            CAMERA_POSE_KEY: jnp.ones((2, 9), dtype=jnp.float32),
        }

        params = enc.init(jax.random.PRNGKey(0), obs)
        out = enc.apply(params, obs)

        assert out.shape == (2, 64)
        assert jnp.isfinite(out).all()


class TestConvEncoderWorldPoints:
    """World-point mode for the shared spatial CNN encoder (3D-53)."""

    def test_world_points_output_shape_518(self):
        enc = ConvEncoder(input_kind="world_points", embed_dim=256)
        rng = jax.random.PRNGKey(0)
        # (B, 3, H, W) metric XYZ world-point map.
        dummy = jnp.zeros((2, 518, 518, 3))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 256)
        assert jnp.isfinite(out).all()

    def test_world_points_handles_metric_xyz_range(self):
        # Values far outside [0, 1] (metric coords) must not blow up (symlog).
        enc = ConvEncoder(input_kind="world_points", embed_dim=32)
        rng = jax.random.PRNGKey(0)
        big = jnp.full((1, 518, 518, 3), 1e3)
        params = enc.init(rng, big)
        out = enc.apply(params, big)
        assert out.shape == (1, 32)
        assert jnp.isfinite(out).all()

    def test_rejects_unknown_input_kind(self):
        enc = ConvEncoder(input_kind="depth")
        with pytest.raises(ValueError, match="input_kind"):
            enc.init(jax.random.PRNGKey(0), jnp.zeros((1, 64, 64, 3)))


def _vggt_replay_buffer(capacity: int) -> ReplayBuffer:
    return ReplayBuffer(capacity=capacity, num_actions=4)


def _transition_frame(action: int, reward: float, done: bool) -> ObservationFrame:
    return ObservationFrame(
        image=np.empty((0,), dtype=np.uint8),
        is_first=False,
        previous_action=action,
        reward=reward,
        done=done,
    )


class TestVGGTFeatureReplayBuffer:
    """Test float32 VGGT feature replay stores and samples correctly."""

    def test_add_and_sample(self):
        buf = _vggt_replay_buffer(capacity=1000)
        for i in range(200):
            features = np.random.randn(FEATURE_DIM).astype(np.float32)
            buf.add(
                ReplayTransition.from_frame(
                    features, _transition_frame(i % 4, 0.1, i % 50 == 49)
                )
            )
        assert buf.size == 200

        batch = buf.sample(batch_size=4, seq_len=16)
        assert batch.obs.shape == (4, 16, FEATURE_DIM)
        assert batch.obs.dtype == jnp.float32
        assert batch.actions.shape == (4, 16, 4)
        assert batch.rewards.shape == (4, 16)
        assert batch.is_first.shape == (4, 16)

    def test_no_normalization(self):
        """VGGT features should NOT be divided by 255."""
        buf = _vggt_replay_buffer(capacity=100)
        features = np.ones(FEATURE_DIM, dtype=np.float32) * 500.0
        buf.add(ReplayTransition.from_frame(features, _transition_frame(0, 0.0, False)))
        # Add enough for sampling
        for _ in range(63):
            buf.add(
                ReplayTransition.from_frame(features, _transition_frame(0, 0.0, False))
            )

        batch = buf.sample(batch_size=1, seq_len=16)
        # Values should be 500.0, not 500/255
        assert float(batch.obs[0, 0, 0]) == pytest.approx(500.0)

    def test_is_first_at_boundaries(self):
        buf = _vggt_replay_buffer(capacity=1000)
        for i in range(100):
            features = np.zeros(FEATURE_DIM, dtype=np.float32)
            buf.add(
                ReplayTransition.from_frame(
                    features, _transition_frame(0, 0.0, i % 20 == 19)
                )
            )

        batch = buf.sample(batch_size=8, seq_len=16)
        # is_first should be 1.0 at t=0 of every sequence
        assert (batch.is_first[:, 0] == 1.0).all()


class TestVGGTAgentInit:
    """Test the composed learner initializes with VGGT encoder."""

    def test_agent_init_vggt(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(FEATURE_DIM,),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)

        assert agent.embed_size == cfg.vggt_embed_dim
        assert "encoder" in agent.params

    def test_agent_act_vggt(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(FEATURE_DIM,),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)

        obs_dict = {
            "features": np.random.randn(FEATURE_DIM).astype(np.float32),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, obs_dict["is_first"], act_key)
        assert 0 <= action < 4

    def test_agent_act_vggt_aggregator_mlp(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt_aggregator_mlp",
            obs_shape=(POOLED_FEATURE_DIM,),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)

        obs_dict = {
            "features": np.random.randn(POOLED_FEATURE_DIM).astype(np.float32),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, obs_dict["is_first"], act_key)
        assert 0 <= action < 4

    def test_agent_act_vggt_agg_token_transformer_reduced_shape(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt_agg_token_transformer",
            obs_shape=(10 * 16,),
            num_actions=4,
            vggt_token_count=10,
            vggt_token_dim=16,
            vggt_token_projection_dim=32,
            vggt_token_transformer_heads=4,
            vggt_token_transformer_layers=1,
            vggt_embed_dim=64,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)
        assert agent.embed_size == 64

        obs_dict = {
            "features": np.random.randn(10 * 16).astype(np.float16),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, obs_dict["is_first"], act_key)
        assert 0 <= action < 4

    def test_agent_act_wp64_cnn_cp_mlp(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt_wp64_cnn_cp_mlp",
            obs_shape={WORLD_POINTS_KEY: (64, 64, 3), CAMERA_POSE_KEY: (9,)},
            num_actions=4,
            vggt_embed_dim=64,
            encoder_depth=4,
            encoder_mults=(2, 2, 2, 2),
            mlp_vggt_hidden=32,
            mlp_vggt_layers=1,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)

        obs_dict = {
            WORLD_POINTS_KEY: np.zeros((64, 64, 3), dtype=np.float16),
            CAMERA_POSE_KEY: np.zeros((9,), dtype=np.float16),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, obs_dict["is_first"], act_key)
        assert 0 <= action < 4

    def test_agent_act_vggt_wp_dense(self):
        # Full wiring smoke for the dense-WP CNN path (3D-53). Small image keeps
        # the conv forward cheap on CPU; ConvEncoder(world_points) is resolution-agnostic.
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt_wp_dense_cnn",
            obs_shape=(70, 70, 3),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)
        assert "encoder" in agent.params

        obs_dict = {
            "features": np.zeros((70, 70, 3), dtype=np.float32),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, obs_dict["is_first"], act_key)
        assert 0 <= action < 4

    def test_mlp_layers_rejected_by_conv_encoders(self):
        # 3D-52 guard: vggt_mlp_layers must not be silently dropped by a conv
        # encoder (cnn or dense-WP). It should fail loud at agent construction.
        from src.r2dreamer.composition import make_learner

        for enc_type, shape in (
            ("cnn", (64, 64, 3)),
            ("vggt_wp_dense_cnn", (70, 70, 3)),
        ):
            cfg = R2DreamerConfig(
                encoder_type=enc_type,
                obs_shape=shape,
                num_actions=4,
                vggt_mlp_layers=3,
            )
            with pytest.raises(ValueError, match="vggt_mlp_layers"):
                make_learner(cfg, jax.random.PRNGKey(0))

    def test_agent_train_step_vggt(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(FEATURE_DIM,),
            num_actions=4,
            batch_size=2,
            seq_len=8,
            imagination_horizon=3,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)

        B, T = cfg.batch_size, cfg.seq_len
        batch = ReplayBatch(
            obs=jnp.zeros((B, T, FEATURE_DIM)),
            actions=jax.nn.one_hot(
                jnp.zeros((B, T), dtype=jnp.int32), cfg.num_actions
            ),
            rewards=jnp.zeros((B, T)),
            is_first=jnp.zeros((B, T)).at[:, 0].set(1.0),
            is_episode_end=jnp.zeros((B, T)),
        )
        rng, train_key = jax.random.split(rng)
        metrics = agent.train_step(batch, train_key)
        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])

    def test_agent_train_step_vggt_aggregator_mlp(self):
        from src.r2dreamer.composition import make_learner

        cfg = R2DreamerConfig(
            encoder_type="vggt_aggregator_mlp",
            obs_shape=(POOLED_FEATURE_DIM,),
            num_actions=4,
            batch_size=2,
            seq_len=8,
            imagination_horizon=3,
        )
        rng = jax.random.PRNGKey(42)
        agent = make_learner(cfg, rng)

        B, T = cfg.batch_size, cfg.seq_len
        batch = ReplayBatch(
            obs=jnp.zeros((B, T, POOLED_FEATURE_DIM)),
            actions=jax.nn.one_hot(
                jnp.zeros((B, T), dtype=jnp.int32), cfg.num_actions
            ),
            rewards=jnp.zeros((B, T)),
            is_first=jnp.zeros((B, T)).at[:, 0].set(1.0),
            is_episode_end=jnp.zeros((B, T)),
        )
        rng, train_key = jax.random.split(rng)
        metrics = agent.train_step(batch, train_key)
        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])
