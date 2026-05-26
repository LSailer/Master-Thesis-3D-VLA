#!/usr/bin/env bash
# Submit the full 3D-26 ablation sweep: 2 encoders × 3 seeds = 6 prod jobs.
#
# Usage:
#   ./submit_offline_ablation_sweep.sh prod [--dry-run]
#
# Reads STEPS / BUFFER_DIR / WANDB_PROJECT / EXTRA_ARGS from the env and
# passes them through to each underlying ./submit_offline_ablation.sh call.
#
# Optional env: WAIT_FOR=<jobid>  -- adds --dependency=afterok:<jobid> to every
# submission, e.g. to chain behind the 3D-25 collection job.

set -euo pipefail

mode="${1:-}"
extra_flag="${2:-}"

if [ "${mode}" != "prod" ]; then
    echo "usage: $0 prod [--dry-run]" >&2
    echo "(non-prod modes don't make sense for a sweep — use submit_offline_ablation.sh directly)" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
single_submit="${script_dir}/submit_offline_ablation.sh"

ids=()
for encoder in wp_cp aggregator; do
    for seed in 0 1 2; do
        echo "================================================================"
        echo "Submitting ${encoder} seed=${seed}"
        echo "================================================================"
        out="$("${single_submit}" "${mode}" "${encoder}" "${seed}" ${extra_flag})"
        echo "${out}"
        if [ "${extra_flag}" != "--dry-run" ]; then
            jobid="$(echo "${out}" | awk '/Submitted batch job/ {print $NF}')"
            if [ -n "${jobid}" ]; then ids+=("${encoder}-s${seed}:${jobid}"); fi
        fi
    done
done

if [ "${extra_flag}" != "--dry-run" ]; then
    echo ""
    echo "================================================================"
    echo "Submitted 6 jobs:"
    for id in "${ids[@]}"; do echo "  ${id}"; done
    echo "================================================================"
fi
