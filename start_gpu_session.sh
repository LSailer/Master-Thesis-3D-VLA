#!/bin/bash
# Start an srun GPU session, optionally inside tmux (survives SSH disconnects)
#
# Usage: ./start_gpu_session.sh [--tmux] [--claude] [session-name] [partition] [time]
#
# Without flags (default):
#   Runs srun directly in the current shell (no persistence)
#
# With --tmux:
#   Wraps srun in a tmux session on the login node
#   After disconnect: tmux attach -t <session-name>
#
# With --claude:
#   Implies --tmux, launches Claude Code on the GPU node

USE_TMUX=false
USE_CLAUDE=false
while [[ "$1" == --* ]]; do
    case "$1" in
        --tmux)   USE_TMUX=true; shift ;;
        --claude) USE_CLAUDE=true; USE_TMUX=true; shift ;;
        *)        echo "Unknown flag: $1"; exit 1 ;;
    esac
done

SESSION="${1:-gpu-work}"
PARTITION="${2:-gpu_h100-dev}"
TIME="${3:-00:30:00}"
PROJECT_DIR="/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA"

# Without tmux: plain srun in current shell
if [[ "$USE_TMUX" == false ]]; then
    echo "Starting srun on $PARTITION ($TIME) without tmux..."
    srun --partition="$PARTITION" -n1 --gres=gpu:1 --time="$TIME" --pty bash
    exit $?
fi

# Already on a compute node: start tmux directly without srun
if [[ "$(hostname)" == uc3n* ]]; then
    echo "Already on compute node $(hostname), starting tmux without srun..."
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' already exists. Attaching..."
        tmux attach -t "$SESSION"
        exit 0
    fi
    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
    if [[ "$USE_CLAUDE" == true ]]; then
        tmux send-keys -t "$SESSION" "claude" Enter
    fi
    tmux attach -t "$SESSION"
    exit 0
fi

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "Starting tmux session '$SESSION' with srun on $PARTITION ($TIME)..."
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
tmux set-option -t "$SESSION" remain-on-exit on

# Build srun command — optionally chain claude after GPU shell starts
if [[ "$USE_CLAUDE" == true ]]; then
    tmux send-keys -t "$SESSION" "srun --partition=$PARTITION -n1 --gres=gpu:1 --time=$TIME --pty bash -c 'cd $PROJECT_DIR && claude'" Enter
else
    tmux send-keys -t "$SESSION" "srun --partition=$PARTITION -n1 --gres=gpu:1 --time=$TIME --pty bash" Enter
fi

echo "Session '$SESSION' started in background."
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl+b d"
tmux attach -t "$SESSION"
