#!/bin/bash
# Run Ralph on BWUniCluster to execute all TDD cycles from the DreamerV3 spec.
# Usage: bash scripts/run_ralph_spec.sh [--resume]
#
# Prerequisites: claude CLI installed, on BWUniCluster login node
# GPU tests auto-wrapped in srun by the agent prompt

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BRANCH="feat/dreamerv3-tdd-cycles"
SPEC="docs/specs/dreamerv3-objectnav-baseline.md"
WORKTREE=".claude/worktrees/tdd-cycles"

# Create worktree if not resuming
if [[ "${1:-}" != "--resume" ]]; then
    # Clean up stale worktree if it exists
    if [ -d "$WORKTREE" ]; then
        echo "Removing existing worktree at $WORKTREE"
        git worktree remove "$WORKTREE" --force 2>/dev/null || true
    fi
    git worktree add -B "$BRANCH" "$WORKTREE" HEAD
    echo "Created worktree on branch $BRANCH at $WORKTREE"
fi

cd "$WORKTREE"

claude --max-turns 200 --print <<'PROMPT'
You are Ralph, executing TDD cycles from the DreamerV3 ObjectNav spec.

## Instructions

1. Read `docs/specs/dreamerv3-objectnav-baseline.md` §7 — all 10 TDD cycles
2. Execute each cycle in dependency order (1-5 any order, then 6 after 5, 7 after 2, 8 after 3+7, 9 after 7, 10 after 2+7+9):
   - Write the failing test from the spec into the specified test file
   - Run the test, confirm it fails (red)
   - Implement the minimal code to pass
   - Run the test, confirm it passes (green)
   - Commit with the exact message from the spec's "Commit" field
3. After all 10 cycles, run Post-TDD static checks:
   uv run pytest tests/test_dreamerv3_shapes.py -v
   uv run mypy src/dreamerv3/ --ignore-missing-imports
   uv run ruff check src/dreamerv3/
   uv run ruff format --check src/dreamerv3/
4. Push branch and create PR targeting main

## GPU routing

- Cycles 1-9: CPU-only, run directly
- Cycle 10 (Habitat integration) + post-TDD smoke/profiling:
  srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 uv run pytest ...
- If no GPU available (e.g. login node only): skip cycle 10, note in PR

## Rules

- ONE test at a time (strict TDD)
- Commit after each green test
- 3 consecutive failures on same test → skip cycle, note in PR
- Never force push
- Never add Co-Authored-By lines to commits
PROMPT
