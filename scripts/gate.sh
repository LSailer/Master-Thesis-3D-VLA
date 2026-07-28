#!/usr/bin/env bash
set -euo pipefail
# Pre-push gate: the only sanctioned way for agents to push and open a PR.
#
#   scripts/gate.sh check   run the checks, report, no push
#   scripts/gate.sh ship    checks; on green push + open/refresh the PR;
#                           after the 3rd consecutive red on a branch, push
#                           once anyway and open a DRAFT PR carrying the
#                           findings (escalation for human review)
#
# Checks run in ONE SLURM CPU allocation (login node stays free):
# basedpyright (type check + explicit-Any baseline) -> pylint ratchet ->
# CPU pytest suite. See scripts/gate/checks.sh.
#
# State lives in .gate/ (gitignored): attempt counter per branch and the
# last check summary. The pylint baseline auto-tightens on green runs; ship
# commits that tightening so the ratchet only ever moves down.

PARTITIONS=dev_cpu,cpu
TIME_LIMIT=00:20:00
MAX_ATTEMPTS=3

cd "$(git rev-parse --show-toplevel)"
mkdir -p .gate output/gate

MODE=${1:-}
if [[ "$MODE" != "check" && "$MODE" != "ship" ]]; then
    echo "usage: scripts/gate.sh {check|ship}" >&2
    exit 2
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$MODE" == "ship" ]]; then
    if [[ "$BRANCH" == "main" || "$BRANCH" == "HEAD" ]]; then
        echo "gate: refusing to ship from '$BRANCH' - use a feature branch." >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "gate: working tree is dirty - commit or stash first." >&2
        exit 1
    fi
fi

COUNTER_FILE=".gate/attempts-${BRANCH//\//_}"
attempts=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)

echo "gate: running checks in a SLURM CPU allocation (dev_cpu,cpu; <=20 min)..."
check_status=0
srun --partition="$PARTITIONS" --cpus-per-task=16 --mem=8G \
    --time="$TIME_LIMIT" --job-name=gate-checks \
    bash scripts/gate/checks.sh || check_status=$?

echo
cat .gate/last-results.md 2>/dev/null || true
echo

if [[ "$MODE" == "check" ]]; then
    exit "$check_status"
fi

pr_body() {
    local verdict=$1
    {
        echo "## Gate"
        echo
        echo "Verdict: $verdict after $((attempts + 1)) run(s) on \`$BRANCH\`."
        echo
        cat .gate/last-results.md 2>/dev/null || echo "(no check summary)"
        echo
        # Surface every pylint-disable comment this branch adds, so the
        # reviewer sees suppressions without hunting through the diff.
        local base disables
        base=$(git merge-base origin/main HEAD)
        disables=$(git diff "$base"...HEAD -- '*.py' \
            | grep '^+' | grep 'pylint: disable' | sed 's/^+//' || true)
        if [[ -n "$disables" ]]; then
            echo "### New pylint-disable comments in this branch"
            echo '```'
            echo "$disables"
            echo '```'
        else
            echo "No new pylint-disable comments in this branch."
        fi
    } >.gate/pr-body.md
}

open_or_update_pr() {
    local draft_flag=$1
    local title
    title=$(git log -1 --pretty=%s)
    local existing
    existing=$(gh pr list --head "$BRANCH" --state open --json number,isDraft \
        --jq '.[0].number' 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        gh pr edit "$existing" --body-file .gate/pr-body.md >/dev/null
        if [[ "$draft_flag" == "ready" ]]; then
            gh pr ready "$existing" >/dev/null 2>&1 || true
        fi
        echo "gate: updated PR #$existing"
    elif [[ "$draft_flag" == "draft" ]]; then
        gh pr create --draft --title "$title" --body-file .gate/pr-body.md
    else
        gh pr create --title "$title" --body-file .gate/pr-body.md
    fi
}

if ((check_status == 0)); then
    # Ratchet may have tightened the pylint baseline during the check.
    if [[ -n "$(git status --porcelain scripts/gate/pylint-baseline.json)" ]]; then
        git add scripts/gate/pylint-baseline.json
        git commit -q -m "chore(gate): tighten pylint baseline"
        echo "gate: committed tightened pylint baseline."
    fi
    pr_body "GREEN"
    git push -u origin "$BRANCH"
    open_or_update_pr ready
    rm -f "$COUNTER_FILE"
    echo "gate: green - pushed and PR is ready for review."
    exit 0
fi

attempts=$((attempts + 1))
echo "$attempts" >"$COUNTER_FILE"

if ((attempts < MAX_ATTEMPTS)); then
    echo "gate: RED (attempt $attempts/$MAX_ATTEMPTS on '$BRANCH')." >&2
    echo "gate: fix the findings above and re-run scripts/gate.sh ship." >&2
    exit 1
fi

# Third consecutive red: one-time escape. Push and open a DRAFT PR carrying
# the findings, then reset the counter - the next pushes face the full gate
# again.
echo "gate: RED for the ${MAX_ATTEMPTS}rd time - escalating as DRAFT PR." >&2
pr_body "ESCALATED (still red after $MAX_ATTEMPTS attempts)"
git push -u origin "$BRANCH"
open_or_update_pr draft
rm -f "$COUNTER_FILE"
echo "gate: draft PR opened with the open findings. Stop here and report" >&2
echo "gate: the findings to Luca instead of retrying." >&2
exit 1
