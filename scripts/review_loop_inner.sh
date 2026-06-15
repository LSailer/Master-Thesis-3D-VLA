#!/bin/bash
# Inner loop for Linear-grounded PR review automation.
#
# Environment (set by review_loop.sh):
#   MERGE_MODE    - "true" lets the Codex review agent merge eligible PRs
#   RUN_ONCE      - "true" runs one polling iteration and exits
#   POLL_SECONDS  - delay between iterations, default 900

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PROJECT_ROOT="$(pwd)"

MERGE_MODE="${MERGE_MODE:-false}"
RUN_ONCE="${RUN_ONCE:-false}"
POLL_SECONDS="${POLL_SECONDS:-900}"

PROMPT_DIR="$PROJECT_ROOT/scripts/agent_loop_prompts"
WORK_DIR="$PROJECT_ROOT/.review"
WORKTREE_ROOT="$WORK_DIR/worktrees"
LOG_DIR="$WORK_DIR/logs"
STATE_DIR="$WORK_DIR/state"
mkdir -p "$WORKTREE_ROOT" "$LOG_DIR" "$STATE_DIR"

CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-medium}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-7200}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
REMOTE="${REMOTE:-origin}"

log() { echo "[review-loop $(date +%H:%M:%S)] $*"; }

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

preflight() {
    require_cmd git
    require_cmd gh
    require_cmd jq
    require_cmd envsubst
    require_cmd timeout
    require_cmd codex
    gh auth status >/dev/null
    codex exec --help >/dev/null
}

linear_keys_from_text() {
    grep -Eio '[A-Z0-9]*[A-Z][A-Z0-9]*-[0-9]+' | tr '[:lower:]' '[:upper:]' | sort -u
}

primary_linear_key_from_body() {
    grep -Eio 'primary[[:space:]_-]*(linear|issue)?[[:space:]:#-]+[A-Z0-9]*[A-Z][A-Z0-9]*-[0-9]+' \
        | grep -Eio '[A-Z0-9]*[A-Z][A-Z0-9]*-[0-9]+' \
        | tr '[:lower:]' '[:upper:]' \
        | sort -u \
        | head -1
}

linear_key_for_pr() {
    local title="$1" branch="$2" body="$3"
    local keys key_count primary

    keys="$(printf '%s\n%s\n' "$title" "$branch" | linear_keys_from_text || true)"
    key_count="$(printf '%s\n' "$keys" | sed '/^$/d' | wc -l | tr -d ' ')"
    if [[ "$key_count" -eq 1 ]]; then
        printf '%s\n' "$keys"
        return 0
    fi

    if [[ "$key_count" -gt 1 ]]; then
        primary="$(printf '%s\n' "$body" | primary_linear_key_from_body || true)"
        if [[ -n "$primary" ]] && grep -qx "$primary" <<<"$keys"; then
            printf '%s\n' "$primary"
            return 0
        fi
    fi

    return 1
}

has_human_requested_changes() {
    local pr="$1"
    gh pr view "$pr" --json reviews \
        --jq '[.reviews[] | select(.state == "CHANGES_REQUESTED" and ((.author.login | test("\\[bot\\]$")) | not))] | length > 0'
}

eligible_prs_json() {
    gh pr list --state open --limit 100 \
        --json number,title,headRefName,isDraft,body,reviewDecision,mergeStateStatus,url \
        | jq -c '[.[] | select(.isDraft | not)] | sort_by(.number)'
}

prepare_worktree() {
    local pr="$1" worktree="$2"

    git fetch "$REMOTE" "$MAIN_BRANCH" --quiet
    git fetch "$REMOTE" "pull/$pr/head:refs/remotes/$REMOTE/pr-$pr" --quiet

    if [[ -d "$worktree/.git" || -f "$worktree/.git" ]]; then
        git -C "$worktree" fetch "$REMOTE" "$MAIN_BRANCH" --quiet
        git -C "$worktree" checkout --quiet "pr-review-$pr"
        git -C "$worktree" reset --hard "$REMOTE/pr-$pr" --quiet
    else
        rm -rf "$worktree"
        git worktree add --quiet -B "pr-review-$pr" "$worktree" "$REMOTE/pr-$pr"
    fi
}

cleanup_worktree_if_done() {
    local worktree="$1" verdict="$2"
    if [[ "$verdict" == MERGED* || "$verdict" == ESCALATED* || "$verdict" == BLOCKED* ]]; then
        git worktree remove --force "$worktree" >/dev/null 2>&1 || true
    fi
}

run_review_agent() {
    local pr="$1" linear_key="$2" worktree="$3" pr_json="$4"
    local safe_key log_file prompt_file verdict_file status_file pr_json_file agent_state_dir

    safe_key="$(tr '/: ' '___' <<<"pr-$pr-$linear_key")"
    agent_state_dir="$worktree/.review-automation"
    mkdir -p "$agent_state_dir"
    prompt_file="$agent_state_dir/prompt.md"
    verdict_file="$agent_state_dir/verdict"
    status_file="$agent_state_dir/status"
    pr_json_file="$agent_state_dir/pr.json"
    log_file="$LOG_DIR/$safe_key.log"
    rm -f "$prompt_file" "$verdict_file" "$status_file" "$pr_json_file"

    printf '%s\n' "$pr_json" > "$pr_json_file"

    PR_NUMBER="$pr" \
    LINEAR_KEY="$linear_key" \
    WORKTREE="$worktree" \
    MERGE_MODE="$MERGE_MODE" \
    VERDICT_FILE="$verdict_file" \
    PR_JSON_FILE="$pr_json_file" \
        envsubst < "$PROMPT_DIR/pr_review_automation.md" > "$prompt_file"

    log "Invoking Codex for PR #$pr ($linear_key), merge_mode=$MERGE_MODE"
    set +e
    timeout "$CODEX_TIMEOUT" codex exec \
        --cd "$worktree" \
        --model "$CODEX_MODEL" \
        --config "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
        --sandbox workspace-write \
        --ask-for-approval never \
        --output-last-message "$status_file" \
        "$(cat "$prompt_file")" \
        > "$log_file" 2>&1
    local exit_code=$?
    set -e

    if [[ "$exit_code" -ne 0 ]]; then
        log "Codex failed for PR #$pr with exit code $exit_code. See $log_file"
        printf 'BLOCKED: codex exited %s\n' "$exit_code" > "$verdict_file"
    fi

    local verdict
    verdict="$(head -1 "$verdict_file" 2>/dev/null || true)"
    if [[ -z "$verdict" ]]; then
        verdict="BLOCKED: missing verdict file"
        printf '%s\n' "$verdict" > "$verdict_file"
    fi

    log "PR #$pr verdict: $verdict"
    cp "$verdict_file" "$STATE_DIR/$safe_key.verdict" 2>/dev/null || true
    cleanup_worktree_if_done "$worktree" "$verdict"

    [[ "$verdict" == MERGED* ]]
}

poll_once() {
    log "Refreshing PR list..."
    local prs merged_this_iteration=false
    prs="$(eligible_prs_json)"

    if [[ "$(jq 'length' <<<"$prs")" -eq 0 ]]; then
        log "No open non-draft PRs."
        return 0
    fi

    while IFS= read -r pr_json; do
        [[ -n "$pr_json" ]] || continue

        local pr title branch body url review_decision linear_key requested_changes worktree
        pr="$(jq -r '.number' <<<"$pr_json")"
        title="$(jq -r '.title' <<<"$pr_json")"
        branch="$(jq -r '.headRefName' <<<"$pr_json")"
        body="$(jq -r '.body // ""' <<<"$pr_json")"
        url="$(jq -r '.url' <<<"$pr_json")"
        review_decision="$(jq -r '.reviewDecision // ""' <<<"$pr_json")"

        if ! linear_key="$(linear_key_for_pr "$title" "$branch" "$body")"; then
            log "Skipping PR #$pr: expected exactly one Linear key in title/branch ($url)"
            continue
        fi

        if [[ "$review_decision" == "CHANGES_REQUESTED" ]]; then
            requested_changes="$(has_human_requested_changes "$pr")"
            if [[ "$requested_changes" == "true" ]]; then
                log "Skipping PR #$pr ($linear_key): human requested changes"
                continue
            fi
        fi

        worktree="$WORKTREE_ROOT/pr-$pr"
        if ! prepare_worktree "$pr" "$worktree"; then
            log "Skipping PR #$pr ($linear_key): failed to prepare worktree"
            printf 'BLOCKED: failed to prepare worktree\n' > "$STATE_DIR/pr-$pr-$linear_key.verdict"
            continue
        fi

        if run_review_agent "$pr" "$linear_key" "$worktree" "$pr_json"; then
            merged_this_iteration=true
            log "Merged one PR this iteration; stopping until next poll."
            break
        fi
    done < <(jq -c '.[]' <<<"$prs")

    if [[ "$merged_this_iteration" != "true" ]]; then
        log "No PR merged this iteration."
    fi
}

preflight

while true; do
    poll_once

    if [[ "$RUN_ONCE" == "true" ]]; then
        log "RUN_ONCE=true; exiting."
        exit 0
    fi

    log "Sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
done
