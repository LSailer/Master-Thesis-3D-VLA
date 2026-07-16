"""Observation adapters bridging env frames to encoder and replay observations."""

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import (
    VGGTFeatureKind,
    VGGTObsAdapter,
    VGGT_FEATURE_DIM,
)
from src.r2dreamer.adapters.hybrid_adapter import (
    HybridObsAdapter,
    HYBRID_FEATURE_DIM,
)
from src.r2dreamer.adapters.house_context_adapter import VGGTHouseContextObsAdapter
from src.r2dreamer.adapters.house_points_adapter import (
    VGGTHybridHousePointsPoseObsAdapter,
    VGGTHousePointsPoseObsAdapter,
)
from src.r2dreamer.adapters.token_adapters import (
    VGGTHouseFullTokenObsAdapter,
    VGGTHouseGlobalEmbeddingObsAdapter,
    VGGTHouseGlobalTokenObsAdapter,
)
