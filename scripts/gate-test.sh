#!/usr/bin/env bash
set -euo pipefail
# Full suite (CPU + GPU + Habitat E2E) as TWO parallel allocations.
# Deliberate deviation from the "targeted" commands.test contract:
# this repo has no separate CI regression, so the gate owns it.
#
# Why split, and why here: measured 2026-07-26, the whole suite takes 21:24 in
# one allocation, of which tests/vggt is 7:04 - every test there loads the real
# VGGT model. Splitting at that boundary cuts wall-clock to the slower half
# (~14 min) and, more importantly, makes a failure self-locating: "vggt is red"
# names the subsystem, which one combined run does not.
#
# Both halves need a GPU. Splitting CPU-only from GPU-only is NOT possible by
# marker: the heavy tests gate on `pytest.mark.skipif(not HAS_CUDA_JAX)` rather
# than the registered `gpu` marker (only 7 of 663 tests carry those), so `-m`
# cannot select them - without a GPU they silently skip and the run looks green.
# GPU-marked tests also live outside tests/vggt (habitat E2E, SPL, the decoder
# probe), so the core half cannot drop its allocation either.
#
# 30 minutes, not 20: the original 20 was a guess that had never run, and the
# combined suite overran it by 84 seconds. 30 is the dev_gpu_h100 ceiling and
# leaves each half better than 2x headroom.
# Comma list: SLURM picks whichever partition is free first.

PARTITIONS=dev_gpu_h100,gpu_h100_short
TIME_LIMIT=00:30:00

# --no-sync: plain `uv run` re-syncs the shared .venv, which on this cluster
# means pulling gigabytes of CUDA wheels - inside srun that is both slow and
# liable to fail outright.
run_part() {
    local name=$1
    shift
    srun --partition="$PARTITIONS" --gres=gpu:1 --time="$TIME_LIMIT" \
        --job-name="gate-$name" \
        bash -c "RUN_HABITAT_E2E=1 COVERAGE_FILE=.coverage.$name \
                 uv run --no-sync pytest $* -q -p no:randomly \
                 --cov=src --cov-report=" \
        >"output/gate/$name.log" 2>&1
}

mkdir -p output/gate .no-mistakes
rm -f .coverage .coverage.core .coverage.vggt coverage.xml

# Parallel, not sequential: the halves share no state, so wall-clock is the
# slower of the two rather than their sum.
run_part core tests/ --ignore=tests/vggt &
core_pid=$!
run_part vggt tests/vggt &
vggt_pid=$!

# `wait` on a known pid returns that job's status; capture both before failing
# so a red core half still reports whether vggt was red too.
core_status=0
vggt_status=0
wait "$core_pid" || core_status=$?
wait "$vggt_pid" || vggt_status=$?

for part in core vggt; do
    echo "----- $part -----"
    tail -n 20 "output/gate/$part.log"
done

# One report from both halves, or diff-cover would count every line the other
# half covered as untested. Cobertura XML, not JSON: diff-cover only parses XML
# and lcov, and handed a coverage.json it dies on "Unknown syntax in lcov
# report" - which is why the original coverage step never produced a number.
uv run --no-sync python -m coverage combine .coverage.core .coverage.vggt
uv run --no-sync python -m coverage xml -o coverage.xml

# Coverage of changed lines feeds the risk score (measured, not blocking).
# uvx, not `uv run`: diff-cover is a reporting tool, not a project dependency.
uvx diff-cover coverage.xml --compare-branch origin/main \
    --json-report .no-mistakes/diff-cover.json || true

if ((core_status != 0 || vggt_status != 0)); then
    echo "FAILED: core=$core_status vggt=$vggt_status (full logs in output/gate/)" >&2
    exit 1
fi
echo "Both halves passed."
