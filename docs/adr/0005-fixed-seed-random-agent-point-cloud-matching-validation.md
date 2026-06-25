# Validate point-cloud matching with fixed-seed random-action replays

Tracked point-cloud matcher comparisons will use the same fixed-seed random-agent action sequence so index-wise, nearest-neighbor, and voxel variants see identical observations. The validation target is not just total cloud growth: it should separate false-positive additions, where already-seen image content creates new tracked points, from true-positive additions, where newly-seen content legitimately expands the global point cloud.

## Grill questions before implementing the metric

- What is the evaluation unit: image pixel/current point, tracked global point, or voxel? Recommended answer: evaluate per current full-resolution point first, then aggregate per frame and per action.
- What is the reference for "already seen" without using the matcher being tested as its own ground truth? Recommended answer: use STOP/revisit frames as high-confidence already-seen cases first, and record action/camera context for later stronger visibility checks.
- What counts as a false-positive addition? Recommended answer: a current point is appended as new even though an already tracked point exists within the chosen geometric tolerance or occupied voxel neighborhood.
- What counts as a true-positive addition? Recommended answer: a current point is appended because no prior tracked point/voxel is within tolerance, especially at newly visible image boundaries after movement.
- How do we compare thresholds fairly? Recommended answer: replay the same fixed action sequence for every matcher and threshold sweep, then report matched rate, new-point rate, false-positive add rate, true-positive add rate, nearest-neighbor distance quantiles, and total tracked-cloud growth.
