#!/bin/bash
# Start a tmux session with srun on H100 GPU
# Usage: ./start_gpu_session.sh [session-name] [partition] [time]
#
# After experiments, run: claude --remote-control "GPU Session"
# Then detach tmux: Ctrl+b d
# Reattach later: tmux attach -t <session-name>

SESSION="${1:-gpu-work}"
PARTITION="${2:-gpu_h100}"
TIME="${3:-24:00:00}"

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
