#!/usr/bin/env bash
set -euo pipefail

BATS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.bats"
BATS_TAG="v1.10.0"

if [ -x "$BATS_DIR/bin/bats" ]; then
    exit 0
fi

git clone --depth 1 --branch "$BATS_TAG" https://github.com/bats-core/bats-core.git "$BATS_DIR"
