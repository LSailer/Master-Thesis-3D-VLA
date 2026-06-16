#!/bin/bash
# Tmux+srun outer wrapper for Linear-grounded PR review automation.
#
# Usage:
#   ./scripts/review_loop.sh [flags] [session-name]
#
# Flags:
#   --merge           Allow the review agent to squash-merge eligible PRs
#                     (default is dry-run / no merge authority)
#   --once            Run one polling iteration and exit
#   --poll-seconds N  Delay between polling iterations (default 900)
#   --time HH:MM:SS   SLURM --time (default 24:00:00)
#   --partition NAME  SLURM partition (default gpu_h100)
#
# Reattach after disconnect:  tmux attach -t <session-name>
# Detach:                     Ctrl+b d

set -euo pipefail

MERGE_MODE=false
RUN_ONCE=false
POLL_SECONDS=900
TIME="24:00:00"
PARTITION="gpu_h100"

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --merge)        MERGE_MODE=true; shift ;;
        --once)         RUN_ONCE=true; shift ;;
        --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        *)              echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

SESSION="${1:-review-loop}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INNER_CMD="cd $PROJECT_DIR && MERGE_MODE=$MERGE_MODE RUN_ONCE=$RUN_ONCE POLL_SECONDS=$POLL_SECONDS bash scripts/review_loop_inner.sh"

if [[ "$(hostname)" == uc3n* ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' exists. Attaching..."
        tmux attach -t "$SESSION"
        exit 0
    fi

    echo "On compute node $(hostname); starting tmux without srun..."
    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
    tmux set-option -t "$SESSION" remain-on-exit on
    tmux send-keys -t "$SESSION" "$INNER_CMD" Enter
    tmux attach -t "$SESSION"
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "Starting tmux session '$SESSION' with srun on $PARTITION ($TIME)..."
echo "Merge mode: $MERGE_MODE"
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
tmux set-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION" \
    "srun --partition=$PARTITION -n1 --gres=gpu:1 --time=$TIME --pty bash -c \"$INNER_CMD\"" Enter

echo "Session '$SESSION' started."
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl+b d"
tmux attach -t "$SESSION"
