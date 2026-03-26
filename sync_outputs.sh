#!/usr/bin/env bash
# Sync outputs from BWUniCluster to local machine
# Usage: ./sync_outputs.sh

REMOTE="ul_hfj15@bwunicluster.scc.kit.edu"
REMOTE_PATH="/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/outputs/"
LOCAL_PATH="/Users/lucamac/Coding/Master-Thesis-3D-VLA/output/"

mkdir -p "$LOCAL_PATH"

echo "Syncing outputs from cluster..."
echo "  Remote: ${REMOTE}:${REMOTE_PATH}"
echo "  Local:  ${LOCAL_PATH}"
echo

rsync -avz --progress "${REMOTE}:${REMOTE_PATH}" "${LOCAL_PATH}"

echo
echo "Done. Files in ${LOCAL_PATH}:"
ls -lh "${LOCAL_PATH}"
