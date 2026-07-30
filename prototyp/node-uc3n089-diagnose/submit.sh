#!/bin/bash
# Submit the probe pinned to one node: ./submit.sh uc3n089
set -euo pipefail
node=${1:?usage: submit.sh <node>}
cd "$(dirname "$0")/../.."
sbatch --nodelist="$node" prototyp/node-uc3n089-diagnose/probe.sbatch
