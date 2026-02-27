#!/bin/bash
# Clone external repos for local exploration (not committed to git)

EXTERNAL="$(dirname "$0")/../external"

clone_and_install() {
  local name=$1
  local url=$2
  if [ ! -d "$EXTERNAL/$name" ]; then
    echo "Cloning $name..."
    git clone "$url" "$EXTERNAL/$name"
  else
    echo "$name already exists, skipping."
  fi
  if [ -f "$EXTERNAL/$name/requirements.txt" ]; then
    echo "Installing $name deps..."
    uv pip install -r "$EXTERNAL/$name/requirements.txt"
  fi
}

if [ $# -ge 2 ]; then
  # Single-repo mode: bash setup_external.sh <name> <url>
  clone_and_install "$1" "$2"
else
  # Default: clone all
  clone_and_install "VGGT"    "https://github.com/facebookresearch/vggt.git"
  clone_and_install "OpenVLA" "https://github.com/openvla/openvla.git"
  # clone_and_install "UNITE" "<url-when-available>"
fi

echo "Done. External repos in external/"
