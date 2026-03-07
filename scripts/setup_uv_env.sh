#!/bin/bash
# Setup uv environment on BWUniCluster.
# Usage:  source scripts/setup_uv_env.sh
#
# Loads HPC modules, installs build deps, then runs uv sync
# to build habitat-sim from source (headless + bullet).

set -euo pipefail

echo "=== Loading BWUniCluster modules ==="
module load devel/cmake/3.27
module load compiler/gnu/14.2
module load devel/cuda/12.4

echo "=== Checking uv ==="
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

echo "=== Installing build deps into venv ==="
uv pip install scikit-build-core pybind11 cmake ninja

echo "=== Setting habitat-sim build flags ==="
export HABITAT_BUILD_GUI_VIEWERS=OFF
export HABITAT_WITH_BULLET=ON
export HABITAT_WITH_CUDA=ON

echo "=== Running uv sync (builds habitat-sim from source) ==="
uv sync

echo "=== Done ==="
echo "Activate with:  source .venv/bin/activate"
