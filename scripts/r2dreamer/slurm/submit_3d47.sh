#!/usr/bin/env bash
# Submit the 3D-47 ablation: aggregator global-only (3072-d) vs frame+global
# (6144-d) as R2Dreamer input. Collects a dedicated buffer that contains
# z_aggregator_both.npz (the 3D-25/26 buffer predates it), then trains the
# variants off it. Mirrors the 3D-26 submit wrappers' style.
#
# Usage:
#   ./submit_3d47.sh smoke [--dry-run]
#   ./submit_3d47.sh prod  [--dry-run]
#
#   smoke -> gpu_h100_short, 25min: tiny real collection (1500 steps) then a
#            200-step aggregator_both train, both chained afterok. Validates the
#            full collect->verify->train path with real VGGT before prod.
#   prod  -> gpu_h100_il: 400k-step collection (~12h) then ENCODERS x SEEDS
#            training (~8h each), all chained afterok the collection.
#
# Env knobs (prod):
#   ENCODERS    space-separated (default "aggregator aggregator_both")
#   SEEDS       space-separated (default "0 1 2")
#   SKIP_COLLECT=1  reuse an existing BUFFER_DIR instead of collecting
#   BUFFER_DIR  prod buffer dir (default data/offline_buffer_3d47)
#   WAIT_FOR    external jobid the collection should wait on (afterok)
#
# Every submit is pre-flighted with `sbatch --test-only` first.
set -euo pipefail

usage() {
    echo "usage: $0 {smoke|prod} [--dry-run]" >&2
    exit 2
}

mode="${1:-}"
shift || usage
dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        *) usage ;;
    esac
    shift
done
case "${mode}" in smoke|prod) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
collect_sbatch="${script_dir}/collect_offline_buffer_3d47.sbatch"
train_sbatch="${script_dir}/train_offline_ablation_3d47.sbatch"

# submit_job <sbatch_file> <dependency_jid_or_empty> -- prints jobid on stdout,
# diagnostics on stderr. Relies on the caller having exported the env vars the
# sbatch reads; uses --export=ALL like the other 3D wrappers.
submit_job() {
    local sbatch_file="$1"; local dep="$2"; shift 2
    local args=(--partition="${PARTITION}" --time="${TIME_LIMIT}"
                --job-name="${JOB_NAME}" --export=ALL)
    if [ -n "${dep}" ]; then
        args+=(--dependency="afterok:${dep}" --kill-on-invalid-dep=yes)
    fi
    args+=("${sbatch_file}")
    sbatch --test-only "${args[@]}" >&2
    if [ "${dry_run}" -eq 1 ]; then
        echo "    [dry-run] would submit ${JOB_NAME} (${PARTITION}, ${TIME_LIMIT}, dep='${dep}')" >&2
        echo "DRYRUN"
        return 0
    fi
    sbatch --parsable "${args[@]}"
}

if [ "${mode}" = "smoke" ]; then
    export PARTITION=gpu_h100_short
    export TIME_LIMIT=00:25:00
    smoke_buf="data/offline_buffer_3d47_smoke"

    # 1) tiny real collection
    export OUT_DIR="${smoke_buf}" N_STEPS="${N_STEPS:-1500}" JOB_NAME=3d47-smoke-collect
    collect_jid="$(submit_job "${collect_sbatch}" "")"
    echo "[smoke collect] jid=${collect_jid}"

    # 2) aggregator_both train, chained afterok the collection
    export ENCODER=aggregator_both SEED=0 STEPS=200 BUFFER_DIR="${smoke_buf}"
    export EXTRA_ARGS="--skip-heldout-eval --checkpoint-every 100 --log-every 10"
    export JOB_NAME=3d47-smoke-train-aggregator_both
    dep="${collect_jid}"; [ "${dep}" = "DRYRUN" ] && dep=""
    train_jid="$(submit_job "${train_sbatch}" "${dep}")"
    echo "[smoke train ] jid=${train_jid} (afterok:${collect_jid})"
    echo "watch: squeue -j ${collect_jid},${train_jid}"
    exit 0
fi

# ---- prod ----
export PARTITION=gpu_h100_il
buffer_dir="${BUFFER_DIR:-data/offline_buffer_3d47}"
encoders="${ENCODERS:-aggregator aggregator_both}"
seeds="${SEEDS:-0 1 2}"

collect_jid=""
if [ "${SKIP_COLLECT:-0}" = "1" ]; then
    echo "[prod] SKIP_COLLECT=1 — reusing existing buffer ${buffer_dir}"
else
    export TIME_LIMIT=12:00:00 OUT_DIR="${buffer_dir}" N_STEPS="${N_STEPS:-400000}"
    export JOB_NAME=3d47-prod-collect
    collect_jid="$(submit_job "${collect_sbatch}" "${WAIT_FOR:-}")"
    echo "[prod collect] jid=${collect_jid}"
fi

export TIME_LIMIT=08:00:00 BUFFER_DIR="${buffer_dir}" STEPS="${STEPS:-500000}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"
for enc in ${encoders}; do
    for s in ${seeds}; do
        export ENCODER="${enc}" SEED="${s}" JOB_NAME="3d47-prod-${enc}-s${s}"
        dep="${collect_jid}"; [ "${dep}" = "DRYRUN" ] && dep=""
        jid="$(submit_job "${train_sbatch}" "${dep}")"
        echo "[prod train ] ${enc} seed${s} jid=${jid}${collect_jid:+ (afterok:${collect_jid})}"
    done
done
