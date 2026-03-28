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
