#!/usr/bin/env bash
# Submit the full 3D-27 online navigation eval sweep: 2 encoders × 3 seeds = 6 jobs.
#
# Usage:
#   ./submit_online_eval_sweep.sh prod [--dry-run]
#
# Reads CHECKPOINT_ROOT / EPISODES / SPLIT / WANDB_PROJECT / EXTRA_ARGS from env
# and passes them through to each underlying ./submit_online_eval.sh call.
#
# Optional env: WAIT_FOR=<jobid>  -- adds --dependency=afterok:<jobid> to every
# submission, e.g. to chain behind a final 3D-26 training job.

set -euo pipefail

mode="${1:-}"
extra_flag="${2:-}"

if [ "${mode}" != "prod" ]; then
    echo "usage: $0 prod [--dry-run]" >&2
    echo "(use submit_online_eval.sh directly for dev smoke tests)" >&2
    exit 2
fi
if [ -n "${extra_flag}" ] && [ "${extra_flag}" != "--dry-run" ]; then
    echo "usage: $0 prod [--dry-run]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
single_submit="${script_dir}/submit_online_eval.sh"

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
