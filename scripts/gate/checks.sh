#!/usr/bin/env bash
set -uo pipefail
# Sequential gate checks with fail-fast, meant to run inside ONE SLURM CPU
# allocation (scripts/gate.sh submits it). Stage order is cheapest-first so a
# red typecheck spares the 6-minute test suite.
#
# Writes a per-stage summary to .gate/last-results.md (consumed by gate.sh
# for the PR body) and full stage logs to output/gate/<stage>.log.
#
# basedpyright is pinned: the baseline in .basedpyright/baseline.json is only
# stable against the version that wrote it.

BASEDPYRIGHT_VERSION=1.39.9

cd "$(dirname "$0")/../.."
mkdir -p .gate output/gate
RESULTS=.gate/last-results.md
: >"$RESULTS"

note() { echo "$*" | tee -a "$RESULTS"; }

run_stage() {
    local name=$1
    shift
    local log="output/gate/$name.log"
    local start end
    start=$(date +%s)
    "$@" >"$log" 2>&1
    local status=$?
    end=$(date +%s)
    if ((status == 0)); then
        note "- $name: OK ($((end - start))s)"
    else
        note "- $name: FAILED ($((end - start))s, exit $status)"
        echo | tee -a "$RESULTS"
        # The tail is the agent's feedback; keep it in the summary.
        tail -n 40 "$log" | tee -a "$RESULTS"
        exit "$status"
    fi
}

typecheck() {
    uvx "basedpyright@$BASEDPYRIGHT_VERSION"
}

lint() {
    uv run --no-sync python -m pylint src -f json --exit-zero \
        | uv run --no-sync python scripts/gate/pylint_ratchet.py \
            --baseline scripts/gate/pylint-baseline.json
}

run_stage typecheck typecheck
run_stage pylint-ratchet lint
run_stage pytest bash scripts/gate-test.sh

note ""
note "All gate checks passed."
