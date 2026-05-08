#!/usr/bin/env bash
# One-time setup: create a dedicated venv for external/r2dreamer.
#
# Why a separate venv? The parent uv project pins numpy>=2.0 (override) and
# torch>=2.7 (range), while external/r2dreamer pins numpy==1.26.0 and
# torch==2.8.0 alongside tensordict==0.9.1, torchrl==0.9.2 — incompatible
# closures. Isolating the external venv is cleaner than rewriting upstream
# pins.
#
# Run from the repo root.
set -euo pipefail

EXT_DIR="external/r2dreamer"
VENV="$EXT_DIR/.venv"

if [ ! -d "$EXT_DIR" ]; then
    echo "ERROR: $EXT_DIR not found. Run from repo root." >&2
    exit 1
fi

if [ -d "$VENV" ]; then
    echo "venv already exists at $VENV — remove it first if you want a clean rebuild."
    exit 0
fi

echo "[setup] creating venv at $VENV (Python 3.10)"
uv venv "$VENV" --python 3.10

# Run uv pip from /tmp so the parent project's pyproject.toml + uv config
# (notably `override-dependencies = ["numpy>=2.0"]`) is NOT applied to the
# dedicated venv. Without this, uv pins numpy==2.2.x, which breaks cv2's
# numpy ABI.
REQ_ABS="$(realpath "$EXT_DIR/requirements.txt")"
VENV_PY="$(realpath "$VENV/bin/python")"

echo "[setup] installing $EXT_DIR/requirements.txt (from /tmp to bypass parent uv overrides)"
( cd /tmp && uv pip install --python "$VENV_PY" -r "$REQ_ABS" )

echo "[setup] adding wandb (for the run_external_crafter.py wrapper)"
( cd /tmp && uv pip install --python "$VENV_PY" wandb )

# Final sanity: verify numpy is the pinned 1.26 and cv2/moviepy import.
echo "[setup] verifying critical imports"
"$VENV_PY" -c "import numpy; assert numpy.__version__.startswith('1.26'), f'numpy {numpy.__version__} != 1.26.x'; import cv2; from moviepy import editor; print(f'OK: numpy {numpy.__version__}, cv2 {cv2.__version__}')"

echo "[setup] done. Activate with: source $VENV/bin/activate"
