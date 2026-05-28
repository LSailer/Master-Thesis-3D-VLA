#!/usr/bin/env bash
# Record HM3D ObjectNav videos for the L1 2M-step CNN checkpoint.
#
# Renders the env at 512x512 for high-fidelity FPV + top-down panels and
# downsamples to 64x64 before the CNN policy (which was trained at 64x64).
# Writes MP4s to output/cnn-2M-512-video/.
#
# Run from the repo root, inside an interactive H100 srun session
# (see AGENTS.md "Verification" for the 30-min dev-GPU policy).

set -euo pipefail

CKPT="${CKPT:-output/r2dreamer-curriculum-l1/run-4367942/checkpoints/step_002000000.pkl}"
EPISODES="${EPISODES:-3}"
OUT_DIR="${OUT_DIR:-output/cnn-2M-512-video}"
CURRICULUM="${CURRICULUM:-L1}"
SPLIT="${SPLIT:-val}"
VIDEO_RES="${VIDEO_RES:-512}"
FPS="${FPS:-10}"

if [ ! -f "${CKPT}" ]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cat <<EOF
Recording CNN ObjectNav video
  checkpoint:        ${CKPT}
  curriculum:        ${CURRICULUM}
  split:             ${SPLIT}
  episodes:          ${EPISODES}
  video resolution:  ${VIDEO_RES}x${VIDEO_RES} (policy sees 64x64)
  output dir:        ${OUT_DIR}
EOF

.venv/bin/python -m src.main evaluate \
    --env habitat \
    --encoder cnn \
    --curriculum "${CURRICULUM}" \
    --checkpoint "${CKPT}" \
    --episodes "${EPISODES}" \
    --split "${SPLIT}" \
    --output_dir "${OUT_DIR}" \
    --render_resolution 64 \
    --video_render_resolution "${VIDEO_RES}" \
    --save_video_path "${OUT_DIR}" \
    --log_video_episodes "${EPISODES}" \
    --video_fps "${FPS}"

echo "Done. MP4s in ${OUT_DIR}/"
ls -la "${OUT_DIR}"/*.mp4 2>/dev/null || true
