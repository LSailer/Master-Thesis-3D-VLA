#!/bin/bash
# Inner loop for the agent dispatcher. Runs inside srun (or compute-node tmux).
# Picks one ready-for-agent issue, implements it, reviews it, pushes, repeats.
# Hard-stops on first failure; drains the queue cleanly otherwise.
#
# Environment (set by agent_loop.sh):
#   PRD          — restrict to children of PRD #N (empty = drain whole queue)
#   MAX_ISSUES   — upper bound on issues processed per run
#   DRY_RUN      — "true" to pick + log + exit without invoking claude

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PRD="${PRD:-}"
MAX_ISSUES="${MAX_ISSUES:-20}"
DRY_RUN="${DRY_RUN:-false}"

PROMPT_DIR="scripts/agent_loop_prompts"
WORK_DIR=".agent_loop"
mkdir -p "$WORK_DIR"

CLAUDE_FLAGS=(--print --dangerously-skip-permissions --max-turns 200 --model claude-opus-4-7)
IMPLEMENT_TIMEOUT=1800   # 30 min per implement pass
REVIEW_TIMEOUT=600       # 10 min per review pass

log()  { echo "[agent-loop $(date +%H:%M:%S)] $*"; }
halt() { local num="$1" reason="$2" tail="${3:-}"
    log "HALT on issue #$num: $reason"
    {
        echo "Loop halted on this issue."
        echo
        echo "**Reason**: $reason"
        if [[ -n "$tail" ]]; then
            echo
            echo "**Last output (truncated)**:"
            echo '```'
            echo "$tail" | tail -n 100
            echo '```'
        fi
    } > "$WORK_DIR/halt-comment.md"
    gh issue comment "$num" --body-file "$WORK_DIR/halt-comment.md" || true
    echo "LOOP HALTED: see issue #$num" >&2
    exit 1
}

pick_next_issue() {
    # JQ filter avoids gh search-index lag. Sort by issue number ASC for FIFO.
    local extra_filter=""
    if [[ -n "$PRD" ]]; then
        # gh search index can be a few minutes behind; for PRD scope we accept
        # that lag because PRD child sets are written in batch by /to-issues.
        gh issue list --label ready-for-agent --state open --limit 100 \
            --search "in:body \"Parent #$PRD\"" \
            --json number,title,body,labels
    else
        gh issue list --label ready-for-agent --state open --limit 100 \
            --json number,title,body,labels
    fi | jq -c '
        [ .[]
          | select(any(.labels[]; .name == "blocked" or .name == "in-progress" or .name == "needs-info") | not)
        ]
        | sort_by(.number)
        | first // empty
    '
}

extract_parent() {
    # Parse the `## Parent` section of the issue body and return the first #N.
    local body="$1"
    sed -n '/^## Parent/,/^## /p' <<<"$body" | grep -oE '#[0-9]+' | head -1 | tr -d '#'
}

count_open_siblings() {
    local prd="$1"
    gh issue list --search "in:body \"Parent #$prd\"" --state open --limit 100 --json number \
        | jq 'length'
}

children_summary() {
    local prd="$1"
    gh issue list --search "in:body \"Parent #$prd\"" --state closed --limit 100 \
        --json number,title \
        | jq -r '.[] | "- Closes #\(.number) — \(.title)"'
}

processed=0
while [[ $processed -lt $MAX_ISSUES ]]; do
    log "Refreshing main branch..."
    git fetch origin main --quiet
    git checkout main --quiet
    git reset --hard origin/main --quiet

    issue_json="$(pick_next_issue)"
    if [[ -z "$issue_json" ]]; then
        log "Queue drained (no ready-for-agent issues left). Exit 0."
        exit 0
    fi

    num="$(jq -r '.number' <<<"$issue_json")"
    title="$(jq -r '.title' <<<"$issue_json")"
    body="$(jq -r '.body // ""' <<<"$issue_json")"
    parent="$(extract_parent "$body" || true)"

    if [[ -n "$parent" ]]; then
        branch="agent/prd-$parent"
    else
        branch="agent/issue-$num"
    fi

    log "Picked #$num: $title  (branch=$branch, parent=${parent:-none})"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY_RUN: would process #$num on $branch"
        processed=$((processed + 1))
        # In dry-run we don't actually take the issue, so to avoid an infinite
        # loop we exit after listing the next one.
        log "DRY_RUN: stopping after first pick."
        exit 0
    fi

    # Check out (or create) the branch off the latest main / existing remote.
    if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
        git fetch origin "$branch" --quiet
        git checkout -B "$branch" "origin/$branch" --quiet
    else
        git checkout -B "$branch" origin/main --quiet
    fi

    gh issue edit "$num" --add-label in-progress --remove-label ready-for-agent

    # IMPLEMENT
    log "Implement pass on #$num..."
    impl_log="$WORK_DIR/implement-$num.log"
    ISSUE_NUM="$num" ISSUE_TITLE="$title" ISSUE_BODY="$body" BRANCH="$branch" \
        envsubst < "$PROMPT_DIR/implement.md" > "$WORK_DIR/implement-$num.prompt"
    if ! timeout "$IMPLEMENT_TIMEOUT" claude "${CLAUDE_FLAGS[@]}" \
            "$(cat "$WORK_DIR/implement-$num.prompt")" \
            > "$impl_log" 2>&1; then
        halt "$num" "implement pass exited non-zero (or timed out at ${IMPLEMENT_TIMEOUT}s)" "$(cat "$impl_log")"
    fi

    # Sanity: implement pass must have produced at least one new commit on the branch.
    new_commits="$(git rev-list --count "origin/main..HEAD")"
    if [[ "$new_commits" -eq 0 ]]; then
        halt "$num" "implement pass left no new commits on $branch" "$(cat "$impl_log")"
    fi

    # REVIEW
    log "Review pass on #$num..."
    rev_log="$WORK_DIR/review-$num.log"
    rm -f "$WORK_DIR/review-verdict"
    ISSUE_NUM="$num" BRANCH="$branch" VERDICT_FILE="$WORK_DIR/review-verdict" \
        envsubst < "$PROMPT_DIR/review.md" > "$WORK_DIR/review-$num.prompt"
    if ! timeout "$REVIEW_TIMEOUT" claude "${CLAUDE_FLAGS[@]}" \
            "$(cat "$WORK_DIR/review-$num.prompt")" \
            > "$rev_log" 2>&1; then
        halt "$num" "review pass exited non-zero (or timed out at ${REVIEW_TIMEOUT}s)" "$(cat "$rev_log")"
    fi
    verdict="$(cat "$WORK_DIR/review-verdict" 2>/dev/null || echo "MISSING")"
    if [[ "$verdict" != APPROVED* ]]; then
        halt "$num" "review verdict not APPROVED (got: $verdict)" "$(cat "$rev_log")"
    fi

    # PUSH
    log "Pushing $branch..."
    git push -u origin "$branch" --quiet || halt "$num" "git push failed" ""

    # CLOSE issue with branch reference
    gh issue close "$num" --comment "Implemented on \`$branch\`. PR will be opened when this PRD's last child closes (or immediately, for orphan issues)."

    # OPEN PR — orphan immediately, PRD only when last sibling closes
    if [[ -z "$parent" ]]; then
        log "Orphan issue: opening PR $branch -> main"
        gh pr create --base main --head "$branch" \
            --title "$title" \
            --body "Closes #$num.

🤖 Generated by agent_loop.sh
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" \
            --label in-review || halt "$num" "gh pr create failed" ""
    else
        open_left="$(count_open_siblings "$parent")"
        if [[ "$open_left" -eq 0 ]]; then
            log "Last sibling of PRD #$parent closed: opening PR $branch -> main"
            prd_title="$(gh issue view "$parent" --json title -q .title)"
            children_md="$(children_summary "$parent")"
            gh pr create --base main --head "$branch" \
                --title "$prd_title" \
                --body "Closes #$parent.

Children implemented:
$children_md

🤖 Generated by agent_loop.sh
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" \
                --label in-review || halt "$num" "gh pr create for PRD #$parent failed" ""
        else
            log "PRD #$parent still has $open_left open child(ren); deferring PR."
        fi
    fi

    processed=$((processed + 1))
    log "Issue #$num done. Processed $processed/$MAX_ISSUES."
done

log "Hit max-issues budget ($MAX_ISSUES). Exit 0."
