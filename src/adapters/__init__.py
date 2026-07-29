"""Observation adapters and the variant registry the run entry point reads.

One row per experiment arm, mapping the name the CLI and the SLURM configs use
to its adapter class. The key is the run vocabulary and names what is observed;
an ``rgb_`` prefix means the appearance channel is part of the observation, and
its absence means the arm is deliberately blind to appearance:

    rgb                        appearance only (the control baseline)
    rgb_pointmap_pose          appearance + per-frame geometry
    pointmap_pose              geometry only
    pointmap_pose_64           geometry only, coarser reduction
    pointmap_dense             geometry only, unpooled, spatial
    rgb_house_voxels(_gnn)     appearance + pose + accumulated house map
    rgb_house_cloud_episodes   appearance + cross-episode cloud
    rgb_global_tokens          appearance + the global half of the VGGT tokens
    rgb_full_tokens            appearance + both halves, full-width tokens
    aggregator_pooled          pooled tokens only, no appearance channel
    aggregator_pooled_b200k    alias of aggregator_pooled, kept for old run ids

Everything the run needs to know about a variant is declared on the adapter
itself, because those are properties of the observation pipeline, not of the
launcher:

    RENDER_RESOLUTION   env frame side length the adapter consumes
    NEEDS_FEATURES      whether the adapter takes a frozen VGGT extractor
    EXTRACTOR_KWARGS    JAXVGGTFeatureExtractor constructor arguments
    ENCODER_OVERRIDES   RoutedCompositeEncoder branch overrides (usually none:
                        the fusion width is implicit in the number of branches)
    RUN_FLAGS           optional: train-CLI flags this variant reads, handed to
                        its constructor as same-named keywords

``RUN_FLAGS`` is the per-run half of ``ENCODER_OVERRIDES``: a knob that is fixed
for a variant is a class constant, a knob that is chosen per run is a flag the
variant claims. Only ``house_voxels`` claims anything today (its PLY dump
schedule), which is why the constant is optional. Claiming is what lets the
launcher stay generic and still be strict - :func:`src.main.make_adapter`
refuses to start a run whose flags belong to a variant it is not running,
instead of silently ignoring them.

A variant that differs only in a constant (a coarser point-map reduction, a
different cloud branch, no appearance channel) is a subclass overriding that
constant, so the registry stays a flat list of names without duplicating
pipeline logic.

Adding a variant is an adapter module plus one row here. There is no
encoder-type string, no shape table and no module-class lookup to keep in sync,
because each adapter declares the routing of its own fields.
"""

from __future__ import annotations

from src.adapters.global_tokens import (
    AggregatorPooledAdapter,
    AggregatorPooledBudget200kAdapter,
    AggregatorPooledCamPoolAdapter,
    AggregatorPooledCamPoolMeanMaxAdapter,
    AggregatorPooledFrameMeanAdapter,
    AggregatorPooledFrameOnlyAdapter,
    AggregatorPooledFullAdapter,
    AggregatorPooledFullDeepAdapter,
    AggregatorPooledFullDeltaAdapter,
    AggregatorPooledFullQuadAdapter,
    AggregatorPooledFullSplitAdapter,
    AggregatorPooledGrid4Adapter,
    AggregatorPooledGridAdapter,
    FullTokensAdapter,
    GlobalTokensAdapter,
)
from src.adapters.house_cloud_episodes import HouseCloudEpisodesAdapter
from src.adapters.house_voxels import HouseVoxelsAdapter, HouseVoxelsGnnAdapter
from src.adapters.pointmap_dense import PointMapDenseAdapter
from src.adapters.pointmap_pose import (
    PointMapPose64OnlyAdapter,
    PointMapPoseAdapter,
    PointMapPoseOnlyAdapter,
)
from src.adapters.rgb import RgbAdapter

ADAPTERS: dict[str, type] = {
    "rgb": RgbAdapter,
    "rgb_pointmap_pose": PointMapPoseAdapter,
    "pointmap_pose": PointMapPoseOnlyAdapter,
    "pointmap_pose_64": PointMapPose64OnlyAdapter,
    "pointmap_dense": PointMapDenseAdapter,
    "rgb_house_voxels": HouseVoxelsAdapter,
    "rgb_house_voxels_gnn": HouseVoxelsGnnAdapter,
    "rgb_house_cloud_episodes": HouseCloudEpisodesAdapter,
    "rgb_global_tokens": GlobalTokensAdapter,
    "rgb_full_tokens": FullTokensAdapter,
    "aggregator_pooled": AggregatorPooledAdapter,
    "aggregator_pooled_b200k": AggregatorPooledBudget200kAdapter,
    "aggregator_pooled_full": AggregatorPooledFullAdapter,
    "aggregator_pooled_meanf": AggregatorPooledFrameMeanAdapter,
    "aggregator_pooled_full_delta": AggregatorPooledFullDeltaAdapter,
    "aggregator_pooled_full_split": AggregatorPooledFullSplitAdapter,
    "aggregator_pooled_full_quad": AggregatorPooledFullQuadAdapter,
    "aggregator_pooled_frame": AggregatorPooledFrameOnlyAdapter,
    "aggregator_pooled_full_deep": AggregatorPooledFullDeepAdapter,
    "aggregator_pooled_campool": AggregatorPooledCamPoolAdapter,
    "aggregator_pooled_campool_meanmax": AggregatorPooledCamPoolMeanMaxAdapter,
    "aggregator_pooled_grid2": AggregatorPooledGridAdapter,
    "aggregator_pooled_grid4": AggregatorPooledGrid4Adapter,
}

__all__ = [
    "ADAPTERS",
    "AggregatorPooledAdapter",
    "AggregatorPooledBudget200kAdapter",
    "AggregatorPooledCamPoolAdapter",
    "AggregatorPooledCamPoolMeanMaxAdapter",
    "AggregatorPooledFrameMeanAdapter",
    "AggregatorPooledFrameOnlyAdapter",
    "AggregatorPooledFullAdapter",
    "AggregatorPooledFullDeepAdapter",
    "AggregatorPooledFullDeltaAdapter",
    "AggregatorPooledFullQuadAdapter",
    "AggregatorPooledFullSplitAdapter",
    "AggregatorPooledGrid4Adapter",
    "AggregatorPooledGridAdapter",
    "FullTokensAdapter",
    "GlobalTokensAdapter",
    "HouseCloudEpisodesAdapter",
    "HouseVoxelsAdapter",
    "HouseVoxelsGnnAdapter",
    "PointMapDenseAdapter",
    "PointMapPose64OnlyAdapter",
    "PointMapPoseAdapter",
    "PointMapPoseOnlyAdapter",
    "RgbAdapter",
]
