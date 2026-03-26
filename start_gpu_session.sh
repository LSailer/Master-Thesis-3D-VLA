#!/bin/bash
# Start srun on H100 GPU, optionally inside a tmux session
# Usage: ./start_gpu_session.sh [--tmux] [session-name] [partition] [time]
#
# Without tmux (default):
#   Runs srun directly in the current shell
#
# With tmux (--tmux):
#   After experiments, run: claude --remote-control "GPU Session"
#   Then detach tmux: Ctrl+b d
#   Reattach later: tmux attach -t <session-name>

USE_TMUX=false
if [[ "$1" == "--tmux" ]]; then
    USE_TMUX=true
    shift
fi

SESSION="${1:-gpu-work}"
PARTITION="${2:-gpu_h100}"
TIME="${3:-24:00:00}"

if [[ "$USE_TMUX" == false ]]; then
    echo "Starting srun on $PARTITION ($TIME) without tmux..."
    srun --partition="$PARTITION" -n1 --gres=gpu:1 --time="$TIME" --pty bash
    exit $?
fi

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "Starting tmux session '$SESSION' with srun on $PARTITION ($TIME)..."
tmux new-session -d -s "$SESSION" \
    "srun --partition=$PARTITION -n1 --gres=gpu:1 --time=$TIME --pty bash"
tmux attach -t "$SESSION"
