#!/usr/bin/env bash
# SessionEnd hook: reflects on the finished session's transcript via a headless
# `claude -p` call and appends a distilled Learning Opportunities entry to
# .claude/LEARNINGS.md. disableAllHooks on the inner call prevents it from
# recursively triggering this same hook.
set -euo pipefail

input=$(cat)
transcript_path=$(printf '%s' "$input" | jq -r '.transcript_path // empty')
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty')
session_id=$(printf '%s' "$input" | jq -r '.session_id // "unknown"')
[ -z "$cwd" ] && cwd="$(pwd)"

[ -z "$transcript_path" ] && exit 0
[ -f "$transcript_path" ] || exit 0

template="$cwd/.claude/templates/FEEDBACK_SUMMARY.md"
learnings="$cwd/.claude/LEARNINGS.md"
[ -f "$template" ] || exit 0

prompt="Read the session transcript at ${transcript_path} (JSONL, one message/event per line) with the Read tool. Using this template as your structure:

$(cat "$template")

Mentally fill in the template based on what happened in the transcript. Then output ONLY a distilled, concise bullet list (2-5 bullets, each a concrete, non-obvious, actionable takeaway for future sessions in this repo) of the Learning Opportunities section. If there is nothing non-obvious worth recording, output exactly: NOTHING_TO_LEARN. Output nothing else — no preamble, no headers, no restatement of the task."

output=$(claude -p "$prompt" \
  --permission-mode dontAsk \
  --allowedTools "Read" \
  --settings '{"disableAllHooks": true}' \
  < /dev/null 2>/dev/null || true)

learning=$(printf '%s' "$output" | sed -e 's/^[[:space:]]*//' -e '/^$/d')

if [ -z "$learning" ] || [ "$learning" = "NOTHING_TO_LEARN" ]; then
  exit 0
fi

{
  printf '\n## %s (session %s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${session_id:0:8}"
  printf '%s\n' "$learning"
} >> "$learnings"

exit 0
