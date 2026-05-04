#!/usr/bin/env bash
# Start the GitHub Actions self-hosted runner for Master-Thesis-3D-VLA in tmux.
#
# Idempotent: if the session already exists, does nothing.
# Self-healing: if GitHub auto-deleted the runner registration (happens after
#               ~14 days inactivity), re-registers with a fresh token.
# Login-node-sticky: tmux is per-host. SSH back to the same login node
#                    (printed below) to reattach.
#
# Reattach:  tmux attach -t thesis-runner
# Detach:    Ctrl-b d
# Stop:      tmux kill-session -t thesis-runner
# Status:    gh api /repos/LSailer/Master-Thesis-3D-VLA/actions/runners
#
# Why a script: Claude Code's sandboxed Bash kills detached children when the
# tool returns, so the runner must be launched from a real interactive shell.

set -euo pipefail

SESSION="thesis-runner"
RUNNER_DIR="$HOME/action_runner_master_thesis"
REPO="LSailer/Master-Thesis-3D-VLA"
RUNNER_NAME="BWUNIcluster"
LOG_FILE="/tmp/${USER}-thesis-runner.log"

# Preflight
[ -d "$RUNNER_DIR" ] || { echo "ERROR: $RUNNER_DIR not found" >&2; exit 1; }
[ -x "$RUNNER_DIR/run.sh" ] || { echo "ERROR: $RUNNER_DIR/run.sh not executable" >&2; exit 1; }
command -v tmux >/dev/null || { echo "ERROR: tmux not on PATH" >&2; exit 1; }
command -v gh   >/dev/null || { echo "ERROR: gh not on PATH"   >&2; exit 1; }

HOST="$(hostname)"

reregister() {
    echo "Registration appears stale — re-registering with fresh token..."
    cd "$RUNNER_DIR"
    local removal_token
    removal_token=$(gh api -X POST "/repos/$REPO/actions/runners/remove-token" --jq '.token' 2>/dev/null || true)
    if [ -n "$removal_token" ]; then
        ./config.sh remove --token "$removal_token" >/dev/null 2>&1 || true
    fi
    rm -f .runner .credentials .credentials_rsaparams .path .env
    local reg_token
    reg_token=$(gh api -X POST "/repos/$REPO/actions/runners/registration-token" --jq '.token')
    [ -n "$reg_token" ] || { echo "ERROR: could not fetch registration token (gh auth scopes?)" >&2; return 1; }
    ./config.sh \
        --url "https://github.com/$REPO" \
        --token "$reg_token" \
        --name "$RUNNER_NAME" \
        --labels self-hosted,Linux,X64 \
        --work _work \
        --unattended >/dev/null
    echo "Re-registered."
    cd - >/dev/null
}

start_session() {
    : > "$LOG_FILE"
    tmux new-session -d -s "$SESSION" -c "$RUNNER_DIR" \
        "exec ./run.sh 2>&1 | tee '$LOG_FILE'"
}

wait_for_online() {
    local seconds=${1:-24}
    local iters=$((seconds / 2))
    for _ in $(seq 1 "$iters"); do
        sleep 2
        local status
        status=$(gh api "/repos/$REPO/actions/runners" \
            --jq ".runners[] | select(.name==\"$RUNNER_NAME\") | .status" 2>/dev/null || true)
        [ "$status" = "online" ] && return 0
    done
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists on $HOST."
else
    start_session
    echo "Started tmux session '$SESSION' on $HOST."
fi

echo -n "Waiting for runner '$RUNNER_NAME' to come online (up to 24 s)"
if wait_for_online 24; then
    echo " — OK"
else
    echo
    # Diagnose: did the listener die because registration is stale?
    if grep -q "registration has been deleted" "$LOG_FILE" 2>/dev/null; then
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        reregister || exit 1
        start_session
        echo -n "Waiting for runner to come online after re-register"
        if wait_for_online 24; then
            echo " — OK"
        else
            echo
            echo "ERROR: still not online after re-register. Inspect: tmux attach -t $SESSION" >&2
            exit 1
        fi
    else
        echo "ERROR: runner not online and no 'registration deleted' marker." >&2
        echo "       Inspect: tmux attach -t $SESSION  (or tail $LOG_FILE)" >&2
        exit 1
    fi
fi

echo
echo "Runner $RUNNER_NAME online (host: $HOST)"
echo "Reattach: ssh ...@${HOST%.localdomain}.scc.kit.edu  ;  tmux attach -t $SESSION"
echo "Stop:     tmux kill-session -t $SESSION"
echo "Log tail: tail -f $LOG_FILE"
