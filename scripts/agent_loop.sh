#!/bin/bash
# Tmux+srun outer wrapper for the agent loop.
#
# Picks `ready-for-agent` issues from GitHub, runs them through Claude
# (implement + review), pushes commits, and opens one PR per PRD (or per
# orphan issue) — all inside a tmux session that survives SSH disconnects.
#
# Usage:
#   ./scripts/agent_loop.sh [flags] [session-name]
#
# Flags:
#   --prd N            Only pick children of PRD #N (omit to drain the whole queue)
#   --max-issues N     Stop after N issues processed (default 20)
#   --time HH:MM:SS    SLURM --time (default 24:00:00)
#   --partition NAME   SLURM partition (default gpu_h100)
#   --dry-run          Pick + log + exit, do not invoke claude
#
# Reattach after disconnect:  tmux attach -t <session-name>
# Detach:                     Ctrl+b d

set -euo pipefail

PRD=""
MAX_ISSUES=20
TIME="24:00:00"
PARTITION="gpu_h100"
DRY_RUN=false

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --prd)        PRD="$2"; shift 2 ;;
        --max-issues) MAX_ISSUES="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        *)            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

SESSION="${1:-agent-loop}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build the inner command that runs inside srun
INNER_CMD="cd $PROJECT_DIR && PRD='$PRD' MAX_ISSUES=$MAX_ISSUES DRY_RUN=$DRY_RUN bash scripts/agent_loop_inner.sh"

# Already on a compute node: skip srun, run tmux directly
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

# Login node: wrap srun in tmux
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "Starting tmux session '$SESSION' with srun on $PARTITION ($TIME)..."
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
tmux set-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION" \
    "srun --partition=$PARTITION -n1 --gres=gpu:1 --time=$TIME --pty bash -c \"$INNER_CMD\"" Enter

echo "Session '$SESSION' started."
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl+b d"
tmux attach -t "$SESSION"
