#!/usr/bin/env bash
# SessionEnd hook: distil the most recent transcript into 0-3 concrete lessons
# and append to docs/wiki/lessons/YYYY-MM-DD.md.
#
# Output format expected from the inner Claude invocation: either "NONE" or
# bullets like "- [gotcha|finding|decision|deadend] one-line lesson - context".
set -euo pipefail

# Resolve project root from this script's physical location so the hook
# works on any host (laptop or cluster). Claude's transcript dir mangles
# both `/` and `_` to `-` in the project path.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PROJECT_SLUG="$(echo "$PROJECT_DIR" | sed 's|[/_]|-|g')"
TRANSCRIPT_DIR="$HOME/.claude/projects/$PROJECT_SLUG"
LESSONS_DIR="$PROJECT_DIR/docs/wiki/lessons"
LESSONS_FILE="$LESSONS_DIR/$(date +%Y-%m-%d).md"

LATEST=$(ls -t "$TRANSCRIPT_DIR"/*.jsonl 2>/dev/null | head -1)
[ -z "$LATEST" ] && exit 0

# Skip tiny transcripts (likely no real work happened)
LINES=$(wc -l < "$LATEST")
[ "$LINES" -lt 20 ] && exit 0

mkdir -p "$LESSONS_DIR"

PROMPT='Read this Claude Code session transcript. Extract 0-3 *concrete* learnings useful to a future Claude working on this thesis project (R2-Dreamer + VGGT, JAX/Flax + PyTorch, HM3D ObjectNav). Worth keeping: gotchas, surprising findings, validated decisions, dead ends. Skip routine work, generic programming knowledge, and anything already obvious from the codebase. Format strictly as markdown bullets: "- [gotcha|finding|decision|deadend] one-line lesson - context". If nothing earned, output exactly NONE on a single line and nothing else.'

SUMMARY=$(claude -p --model claude-haiku-4-5 "$PROMPT" < "$LATEST" 2>/dev/null || echo "")

# Trim whitespace
SUMMARY=$(echo "$SUMMARY" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

[ -z "$SUMMARY" ] && exit 0
[ "$SUMMARY" = "NONE" ] && exit 0

SHORT_ID=$(basename "$LATEST" .jsonl | head -c 8)
{
  echo ""
  echo "## $(date +%H:%M) - session $SHORT_ID"
  echo "$SUMMARY"
} >> "$LESSONS_FILE"

exit 0
