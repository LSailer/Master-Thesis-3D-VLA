#!/usr/bin/env bash
# Submit the 3D-25 offline buffer collection job in dev or prod mode.
#
# Usage:
#   ./submit_offline_buffer.sh dev  [checkpoint] [--dry-run]
#   ./submit_offline_buffer.sh prod [checkpoint] [--dry-run]
#
# dev  -> dev_gpu_h100, 00:30:00  (smoke test; expected to TIMEOUT)
# prod -> gpu_h100_il,  12:00:00  (full 500k-step collection run)
#
# Pre-flights every submit with `sbatch --test-only`, so an invalid partition,
# bad resource ask, or missing checkpoint fails before queueing.
set -euo pipefail

usage() {
    echo "usage: $0 {dev|prod} [checkpoint] [--dry-run]" >&2
    exit 2
}

mode="${1:-}"
shift || usage

case "${mode}" in
    dev)
        partition=dev_gpu_h100
        time_limit=00:30:00
        job_name=3d25-offline-buffer-dev
        ;;
    prod)
        partition=gpu_h100_il
        time_limit=12:00:00
        job_name=3d25-offline-buffer-prod
        ;;
    *)
        usage
        ;;
esac

dry_run=0
ckpt_arg=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        -*)        usage ;;
        *)
            if [ -n "${ckpt_arg}" ]; then usage; fi
            ckpt_arg="$1"
            ;;
    esac
    shift
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_file="${script_dir}/collect_offline_buffer_3d25.sbatch"

git_common_dir="$(git rev-parse --git-common-dir)"
main_root="$(realpath "${git_common_dir}/..")"
DEFAULT_CKPT="output/r2dreamer-curriculum-l1/run-4367942/checkpoints/step_001000000.pkl"
ckpt="${ckpt_arg:-${DEFAULT_CKPT}}"
if [ ! -f "${ckpt}" ] && [ -f "${main_root}/${ckpt}" ]; then
    ckpt="${main_root}/${ckpt}"
fi
if [ ! -f "${ckpt}" ]; then
    echo "Checkpoint not found in worktree or main checkout: ${ckpt}" >&2
    exit 2
fi

echo "Mode=${mode}  Partition=${partition}  Time=${time_limit}"
echo "Checkpoint=${ckpt}"
echo "Sbatch=${sbatch_file}"

sbatch_args=(
    --partition="${partition}"
    --time="${time_limit}"
    --job-name="${job_name}"
    "${sbatch_file}"
    "${ckpt}"
)

sbatch --test-only "${sbatch_args[@]}"
echo "Pre-flight OK."

if [ "${dry_run}" -eq 1 ]; then
    echo "--dry-run: not submitting."
    exit 0
fi

submit_out="$(sbatch "${sbatch_args[@]}")"
echo "${submit_out}"
jobid="$(echo "${submit_out}" | awk '{print $NF}')"
echo "Log: output/offline-buffer-3d25/slurm-${jobid}.out"
