#!/bin/bash
# Download ALFRED dataset from AWS S3.
# Usage: bash scripts/download_alfred.sh [json|json_feat|full]
#   json      — annotations only (default, ~35 MB)
#   json_feat — annotations + ResNet features (~17 GB)
#   full      — full dataset with images (~100+ GB)
#
# Requires: curl, 7z (p7zip / 7zip)
# Output: data/alfred/json_2.1.0/  (or json_feat_2.1.0/ or full_2.1.0/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data/alfred"

MODE="${1:-json}"

mkdir -p "$DATA_DIR"

# Check for 7z
if ! command -v 7z &>/dev/null; then
  echo "Error: 7z not found. Install with: brew install p7zip  (macOS) or apt install p7zip-full (Linux)"
  exit 1
fi

BASE_URL="https://ai2-vision-alfred.s3-us-west-2.amazonaws.com"

case "$MODE" in
  json)      ARCHIVE_NAME="json_2.1.0" ;;
  json_feat) ARCHIVE_NAME="json_feat_2.1.0" ;;
  full)      ARCHIVE_NAME="full_2.1.0" ;;
  *)
    echo "Usage: bash scripts/download_alfred.sh [json|json_feat|full]"
    exit 1
    ;;
esac

DEST="$DATA_DIR/$ARCHIVE_NAME"
ARCHIVE="$DATA_DIR/${ARCHIVE_NAME}.7z"

if [ -d "$DEST" ]; then
  echo "$ARCHIVE_NAME already exists at $DEST, skipping."
  exit 0
fi

echo "Downloading $ARCHIVE_NAME..."
curl -fSL -o "$ARCHIVE" "$BASE_URL/${ARCHIVE_NAME}.7z"
echo "Extracting..."
7z x "$ARCHIVE" -o"$DATA_DIR" -y
rm "$ARCHIVE"
echo "Done. ALFRED data in $DEST/"
