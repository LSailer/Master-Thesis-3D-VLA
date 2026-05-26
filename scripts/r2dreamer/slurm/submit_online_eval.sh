#!/usr/bin/env bash
# Submit a single 3D-27 online navigation evaluation run for a 3D-26 checkpoint.
#
# Usage:
#   ./submit_online_eval.sh dev  <encoder> <seed> [--dry-run]
#   ./submit_online_eval.sh prod <encoder> <seed> [--dry-run]
#
# encoder ∈ {wp_cp, aggregator}; seed ∈ {0, 1, 2}
#
# Env passthrough:
#   CHECKPOINT       explicit checkpoint path for this run (optional)
#   CHECKPOINT_ROOT  default output/3d26-offline-ablation
#   EPISODES         dev default 2, prod default 50
#   EVAL_SEED        default SEED
#   SPLIT            default val
#   WANDB_PROJECT    optional W&B project for eval videos
#   EXTRA_ARGS       extra CLI flags for src.main evaluate

set -euo pipefail

usage() {
    echo "usage: $0 {dev|prod} <encoder> <seed> [--dry-run]" >&2
    echo "       encoder ∈ {wp_cp, aggregator}; seed ∈ {0, 1, 2}" >&2
    exit 2
}

mode="${1:-}"
encoder="${2:-}"
seed="${3:-}"
shift 3 2>/dev/null || usage
[ -z "${mode}" ] || [ -z "${encoder}" ] || [ -z "${seed}" ] && usage
case "${encoder}" in wp_cp|aggregator) ;; *) usage ;; esac
case "${seed}" in 0|1|2) ;; *) usage ;; esac

dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        *) usage ;;
    esac
    shift
done

case "${mode}" in
    dev)
        partition=gpu_h100_short
        time_limit=00:25:00
        export EPISODES="${EPISODES:-2}"
        export EXTRA_ARGS="${EXTRA_ARGS:---log_video_episodes 0}"
        job_name="3d27-dev-${encoder}-s${seed}"
        ;;
    prod)
        partition=gpu_h100_il
        time_limit=04:00:00
        export EPISODES="${EPISODES:-50}"
        export EXTRA_ARGS="${EXTRA_ARGS:-}"
        job_name="3d27-prod-${encoder}-s${seed}"
        ;;
    *)
        usage
        ;;
esac

export ENCODER="${encoder}"
export SEED="${seed}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_file="${script_dir}/evaluate_offline_ablation.sbatch"

echo "Mode=${mode}  Partition=${partition}  Time=${time_limit}"
echo "Encoder=${encoder}  Seed=${seed}  Episodes=${EPISODES}"
echo "Sbatch=${sbatch_file}"

sbatch_args=(
    --partition="${partition}"
    --time="${time_limit}"
    --job-name="${job_name}"
    --export=ALL
)
if [ -n "${WAIT_FOR:-}" ]; then
    sbatch_args+=(--dependency="afterok:${WAIT_FOR}")
fi
sbatch_args+=("${sbatch_file}")

sbatch --test-only "${sbatch_args[@]}"
echo "Pre-flight OK."

if [ "${dry_run}" -eq 1 ]; then
    echo "--dry-run: not submitting."
    exit 0
fi

submit_out="$(sbatch "${sbatch_args[@]}")"
echo "${submit_out}"
jobid="$(echo "${submit_out}" | awk '{print $NF}')"
echo "Log: output/3d27-online-eval/slurm-${jobid}.out"
