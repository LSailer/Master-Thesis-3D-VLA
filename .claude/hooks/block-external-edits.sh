#!/usr/bin/env bash
# PreToolUse hook: block Edit/Write to external/ directory
# Exit 0 = allow, Exit 2 = block

input=$(cat)
tool_name=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

[[ "$tool_name" != "Edit" && "$tool_name" != "Write" ]] && exit 0

file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if echo "$file_path" | grep -q "/external/"; then
  echo "BLOCKED: cannot modify files in external/ — these are upstream third-party repos" >&2
  exit 2
fi

exit 0
