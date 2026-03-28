import pytest
from src.dreamerv3.r2dreamer_config import R2DreamerConfig


class TestR2DreamerConfig:
    def test_defaults(self):
        cfg = R2DreamerConfig()
        assert cfg.stoch_size == 32 * 16  # 512
        assert cfg.feat_size == 2048 + 512  # 2560
        assert cfg.deter_size == 2048

    def test_size25m(self):
        cfg = R2DreamerConfig.size25M()
        assert cfg.deter_size == 3072
        assert cfg.hidden_size == 384


import jax
import jax.numpy as jnp
from src.dreamerv3.r2dreamer_networks import RMSNorm, BlockLinear

@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)

class TestRMSNorm:
    def test_output_shape(self, rng):
        norm = RMSNorm()
        x = jnp.ones((4, 256))
        params = norm.init(rng, x)
        out = norm.apply(params, x)
        assert out.shape == (4, 256)

    def test_normalizes(self, rng):
        norm = RMSNorm()
        x = jnp.array([[3.0, 4.0]])
        params = norm.init(rng, x)
        out = norm.apply(params, x)
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-4)
        expected = x / rms  # scale init = 1
        assert jnp.allclose(out, expected, atol=1e-5)

class TestBlockLinear:
    def test_output_shape(self, rng):
        bl = BlockLinear(out_features=512, blocks=8)
        x = jnp.zeros((2, 2048))
        params = bl.init(rng, x)
        out = bl.apply(params, x)
        assert out.shape == (2, 512)

    def test_3d_input(self, rng):
        bl = BlockLinear(out_features=256, blocks=8)
        x = jnp.zeros((2, 10, 512))
        params = bl.init(rng, x)
        out = bl.apply(params, x)
        assert out.shape == (2, 10, 256)

    def test_weight_shape(self, rng):
        bl = BlockLinear(out_features=512, blocks=8)
        x = jnp.zeros((1, 2048))
        params = bl.init(rng, x)
        assert params["params"]["kernel"].shape == (64, 256, 8)
        assert params["params"]["bias"].shape == (512,)


from src.dreamerv3.r2dreamer_networks import Deter

class TestDeter:
    def test_output_shape(self, rng):
        cfg = R2DreamerConfig()
        deter_mod = Deter(
            deter_size=cfg.deter_size, stoch_size=cfg.stoch_size,
            act_dim=cfg.num_actions, hidden=cfg.hidden_size,
            blocks=cfg.blocks, dyn_layers=cfg.dyn_layers,
        )
        h = jnp.zeros((2, cfg.deter_size))
        z = jnp.zeros((2, cfg.stoch_size))
        a = jnp.zeros((2, cfg.num_actions))
        params = deter_mod.init(rng, z, h, a)
        h_new = deter_mod.apply(params, z, h, a)
        assert h_new.shape == (2, cfg.deter_size)

    def test_deterministic_with_same_input(self, rng):
        cfg = R2DreamerConfig()
        deter_mod = Deter(
            deter_size=cfg.deter_size, stoch_size=cfg.stoch_size,
            act_dim=cfg.num_actions, hidden=cfg.hidden_size,
            blocks=cfg.blocks, dyn_layers=cfg.dyn_layers,
        )
        h = jnp.ones((1, cfg.deter_size)) * 0.1
        z = jnp.ones((1, cfg.stoch_size)) * 0.1
        a = jnp.zeros((1, cfg.num_actions))
        params = deter_mod.init(rng, z, h, a)
        h1 = deter_mod.apply(params, z, h, a)
        h2 = deter_mod.apply(params, z, h, a)
        assert jnp.allclose(h1, h2)


from src.dreamerv3.r2dreamer_networks import R2RSSM

class TestR2RSSM:
    def test_posterior_step(self, rng):
        cfg = R2DreamerConfig()
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B = 2
        stoch = jnp.zeros((B, cfg.stoch_classes, cfg.stoch_discrete))
        deter = jnp.zeros((B, cfg.deter_size))
        action = jnp.zeros((B, cfg.num_actions))
        embed_dim = cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4  # 16*4*4*4=1024
        embed = jnp.zeros((B, embed_dim))

        params = rssm.init(rng, stoch, deter, action, embed)
        new_stoch, new_deter, post_logit = rssm.apply(
            params, stoch, deter, action, embed)

        assert new_deter.shape == (B, cfg.deter_size)
        assert new_stoch.shape == (B, cfg.stoch_classes, cfg.stoch_discrete)
        assert post_logit.shape == (B, cfg.stoch_classes, cfg.stoch_discrete)

    def test_prior_step(self, rng):
        cfg = R2DreamerConfig()
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B = 2
        stoch = jnp.zeros((B, cfg.stoch_classes, cfg.stoch_discrete))
        deter = jnp.zeros((B, cfg.deter_size))
        action = jnp.zeros((B, cfg.num_actions))
        embed_dim = cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4
        embed = jnp.zeros((B, embed_dim))
        params = rssm.init(rng, stoch, deter, action, embed)

        new_stoch, new_deter = rssm.apply(
            params, stoch, deter, action, method=rssm.img_step)
        assert new_deter.shape == (B, cfg.deter_size)
        assert new_stoch.shape == (B, cfg.stoch_classes, cfg.stoch_discrete)

    def test_observe_rollout(self, rng):
        cfg = R2DreamerConfig()
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B, T = 2, 10
        embed_dim = cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4
        embed = jnp.zeros((B, T, embed_dim))
        actions = jnp.zeros((B, T, cfg.num_actions))
        is_first = jnp.zeros((B, T))
        stoch0 = jnp.zeros((B, cfg.stoch_classes, cfg.stoch_discrete))
        deter0 = jnp.zeros((B, cfg.deter_size))

        params = rssm.init(rng, stoch0, deter0, actions[:, 0], embed[:, 0])

        stochs, deters, logits = rssm.apply(
            params, embed, actions, (stoch0, deter0), is_first,
            method=rssm.observe)
        assert stochs.shape == (B, T, cfg.stoch_classes, cfg.stoch_discrete)
        assert deters.shape == (B, T, cfg.deter_size)
        assert logits.shape == (B, T, cfg.stoch_classes, cfg.stoch_discrete)

    def test_get_feat(self):
        cfg = R2DreamerConfig()
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
        )
        B = 2
        stoch = jnp.zeros((B, cfg.stoch_classes, cfg.stoch_discrete))
        deter = jnp.zeros((B, cfg.deter_size))
        feat = rssm.get_feat(stoch, deter)
        assert feat.shape == (B, cfg.stoch_size + cfg.deter_size)


from src.dreamerv3.r2dreamer_networks import R2Encoder, Projector, ReturnEMA

class TestR2Encoder:
    def test_output_shape(self, rng):
        cfg = R2DreamerConfig()
        enc = R2Encoder(depth=cfg.encoder_depth, kernel_size=cfg.encoder_kernel)
        obs = jnp.zeros((2, *cfg.obs_shape))  # (2, 3, 64, 64)
        params = enc.init(rng, obs)
        out = enc.apply(params, obs)
        assert out.shape[0] == 2
        assert out.ndim == 2
        # 16*4 * 4*4 = 1024
        expected_dim = cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4
        assert out.shape[1] == expected_dim

class TestProjector:
    def test_output_shape(self, rng):
        cfg = R2DreamerConfig()
        embed_dim = cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4
        proj = Projector(embed_dim)
        feat = jnp.zeros((2, cfg.feat_size))
        params = proj.init(rng, feat)
        out = proj.apply(params, feat)
        assert out.shape == (2, embed_dim)

    def test_no_bias(self, rng):
        proj = Projector(128)
        x = jnp.zeros((1, 256))
        params = proj.init(rng, x)
        # Should have kernel but no bias
        assert "kernel" in params["params"]["proj"]
        assert "bias" not in params["params"]["proj"]

class TestReturnEMA:
    def test_update_and_stats(self):
        ema = ReturnEMA(alpha=0.5)
        state = ema.init_state()
        returns = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = ema.update(state, returns)
        offset, scale = ema.get_stats(state)
        assert scale >= 1.0
        assert jnp.isfinite(offset)
        assert jnp.isfinite(scale)
