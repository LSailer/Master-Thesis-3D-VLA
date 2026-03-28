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
