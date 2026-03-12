#!/bin/bash
# Run Ralph to process all ready + in-progress AFK issues.
# Usage: bash scripts/run_ralph.sh [max_tasks] [time_limit_min]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MAX_TASKS="${1:-10}"
TIME_LIMIT="${2:-180}"

claude --agent-name ralph --max-turns 200 --print \
  "Execute /ralph-loop with max_tasks=$MAX_TASKS and time_limit=$TIME_LIMIT. Pick up all AFK issues labeled ready or in-progress, implement via TDD, create PRs."
