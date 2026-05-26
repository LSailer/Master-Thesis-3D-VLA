#!/usr/bin/env bash
# Submit one 3D-46 external (PyTorch) R2Dreamer offline run.
#
# Usage:
#   ./submit_external_offline.sh {dev|prod|smoke} <seed> [encoder] [--dry-run]
#
# dev/smoke -> gpu_h100_short (<=30 min), STEPS=200 — sanity / smoke.
# prod      -> gpu_h100_il 8h, STEPS=500000 — the actual 3D-46 baseline runs.
#
# seed    in {0, 1, 2}; encoder defaults wp_cp (3D-46 is WP/CP only).
#
# Launch all three prod seeds:
#   for s in 0 1 2; do ./submit_external_offline.sh prod "$s"; done

set -euo pipefail

usage() {
    echo "usage: $0 {dev|prod|smoke} <seed> [encoder] [--dry-run]" >&2
    echo "       seed in {0,1,2}; encoder in {wp_cp,aggregator} (default wp_cp)" >&2
    exit 2
}

mode="${1:-}"
seed="${2:-}"
[ -z "${mode}" ] || [ -z "${seed}" ] && usage
shift 2 || usage

# Remaining args (any order): optional encoder + optional --dry-run.
encoder="wp_cp"
dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)        dry_run=1 ;;
        wp_cp|aggregator) encoder="$1" ;;
        *)                usage ;;
    esac
    shift
done

case "${seed}"    in 0|1|2)            ;; *) usage ;; esac
case "${encoder}" in wp_cp|aggregator) ;; *) usage ;; esac

case "${mode}" in
    dev|smoke)
        # gpu_h100_short: fast queue, 30-min cap — best fit for smoke. Trim the
        # held-out eval to a few batches so the short window is enough.
        partition=gpu_h100_short
        time_limit=00:25:00
        export STEPS="${STEPS:-200}"
        export EXTRA_ARGS="${EXTRA_ARGS:---checkpoint-every 100 --log-every 10 --heldout-eval-batches 4}"
        # Namespace smoke runs (output + W&B name + tag) so they never collide
        # with the prod runs the comparison globs over.
        export RUN_TAG="${RUN_TAG:-smoke}"
        job_name="3d46-${mode}-${encoder}-s${seed}"
        ;;
    prod)
        partition=gpu_h100_il
        # ~29 steps/s on H100 -> ~5h for 500k; budget 8h for the final held-out
        # eval (non-compiled forward over up to 64 batches).
        time_limit=08:00:00
        export STEPS="${STEPS:-500000}"
        export EXTRA_ARGS="${EXTRA_ARGS:-}"
        job_name="3d46-prod-${encoder}-s${seed}"
        ;;
    *)
        usage
        ;;
esac

export ENCODER="${encoder}"
export SEED="${seed}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_file="${script_dir}/train_external_offline.sbatch"

echo "Mode=${mode}  Partition=${partition}  Time=${time_limit}"
echo "Encoder=${encoder}  Seed=${seed}  Steps=${STEPS}"
echo "Sbatch=${sbatch_file}"

sbatch_args=(
    --partition="${partition}"
    --time="${time_limit}"
    --job-name="${job_name}"
    --export=ALL
)
# Optional: launch only after this job ID completes successfully.
if [ -n "${WAIT_FOR:-}" ]; then
    sbatch_args+=(--dependency="afterok:${WAIT_FOR}")
fi
sbatch_args+=("${sbatch_file}")

# Pre-flight — catches partition / resource mistakes before queueing.
sbatch --test-only "${sbatch_args[@]}"
echo "Pre-flight OK."

if [ "${dry_run}" -eq 1 ]; then
    echo "--dry-run: not submitting."
    exit 0
fi

submit_out="$(sbatch "${sbatch_args[@]}")"
echo "${submit_out}"
jobid="$(echo "${submit_out}" | awk '{print $NF}')"
echo "Log: output/3d46-external-offline/slurm-${jobid}.out"
