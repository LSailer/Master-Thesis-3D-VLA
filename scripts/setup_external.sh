#!/bin/bash
# Clone external repos for local exploration (not committed to git)

EXTERNAL="$(dirname "$0")/../external"

clone_if_missing() {
  local name=$1
  local url=$2
  if [ ! -d "$EXTERNAL/$name" ]; then
    echo "Cloning $name..."
    git clone "$url" "$EXTERNAL/$name"
  else
    echo "$name already exists, skipping."
  fi
}

clone_if_missing "VGGT"    "https://github.com/facebookresearch/vggt.git"
clone_if_missing "OpenVLA" "https://github.com/openvla/openvla.git"
# clone_if_missing "UNITE" "<url-when-available>"

echo "Done. External repos in external/"
