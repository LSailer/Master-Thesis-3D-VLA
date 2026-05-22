#!/usr/bin/env bash
# Submit a single 3D-26 offline R2Dreamer ablation run.
#
# Usage:
#   ./submit_offline_ablation.sh dev    <encoder> <seed> [--dry-run]
#   ./submit_offline_ablation.sh prod   <encoder> <seed> [--dry-run]
#   ./submit_offline_ablation.sh smoke  <encoder> <seed> [--dry-run]
#
# dev/smoke -> dev_gpu_h100 (short, for sanity / smoke tests)
# prod      -> gpu_h100_il 4h (the actual ablation runs)
#
# encoder ∈ {wp_cp, aggregator}
# seed    ∈ {0, 1, 2}
#
# All other knobs (steps, buffer dir, wandb project) come from env vars
# already expected by train_offline_ablation.sbatch — defaults baked in.

set -euo pipefail

usage() {
    echo "usage: $0 {dev|prod|smoke} <encoder> <seed> [--dry-run]" >&2
    echo "       encoder ∈ {wp_cp, aggregator}; seed ∈ {0, 1, 2}" >&2
    exit 2
}

mode="${1:-}"
encoder="${2:-}"
seed="${3:-}"
shift 3 2>/dev/null || usage
[ -z "${mode}" ] || [ -z "${encoder}" ] || [ -z "${seed}" ] && usage

case "${encoder}" in wp_cp|aggregator) ;; *) usage ;; esac
case "${seed}"    in 0|1|2)            ;; *) usage ;; esac

case "${mode}" in
    dev|smoke)
        # gpu_h100_short: 12 nodes, 30-min max. Best fit for smoke — fast queue,
        # doesn't burn the prod gpu_h100_il queue. Avoid dev_gpu_h100: its
        # single node (uc3n082) has been observed to throw
        # CUDA_ERROR_LAUNCH_TIMEOUT during JAX random init for this codebase.
        partition=gpu_h100_short
        time_limit=00:25:00
        export STEPS="${STEPS:-200}"
        export EXTRA_ARGS="${EXTRA_ARGS:---heldout-eval-every 0 --checkpoint-every 100 --log-every 10}"
        job_name="3d26-${mode}-${encoder}-s${seed}"
        ;;
    prod)
        partition=gpu_h100_il
        # Smoke at 27-29 fps -> ~5h for 500k steps; budget 8h for headroom +
        # final heldout eval (compute_heldout_metrics is non-JIT and slow).
        time_limit=08:00:00
        export STEPS="${STEPS:-500000}"
        export EXTRA_ARGS="${EXTRA_ARGS:-}"
        job_name="3d26-prod-${encoder}-s${seed}"
        ;;
    *)
        usage
        ;;
esac

dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        *) usage ;;
    esac
    shift
done

export ENCODER="${encoder}"
export SEED="${seed}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_file="${script_dir}/train_offline_ablation.sbatch"

echo "Mode=${mode}  Partition=${partition}  Time=${time_limit}"
echo "Encoder=${encoder}  Seed=${seed}  Steps=${STEPS}"
echo "Sbatch=${sbatch_file}"

# Pre-flight first — catches partition / resource mistakes before queueing.
sbatch_args=(
    --partition="${partition}"
    --time="${time_limit}"
    --job-name="${job_name}"
    --export=ALL
    "${sbatch_file}"
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
echo "Log: output/3d26-offline-ablation/slurm-${jobid}.out"
