#!/usr/bin/env bash
# SessionStart hook: injects accumulated .claude/LEARNINGS.md into context.
set -euo pipefail

input=$(cat)
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty')
[ -z "$cwd" ] && cwd="$(pwd)"

learnings="$cwd/.claude/LEARNINGS.md"

if [ -s "$learnings" ]; then
  jq -n --rawfile content "$learnings" \
    '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: ("Accumulated learnings from past sessions (.claude/LEARNINGS.md):\n\n" + $content)}}'
else
  echo '{}'
fi
