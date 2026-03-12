#!/usr/bin/env bash
# PostToolUse hook: auto-format Python files

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

# Only format .py files
[[ "$file_path" != *.py ]] && exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

# Try uv first, then bare ruff
if command -v uv &>/dev/null && uv run ruff format "$file_path" 2>/dev/null; then
  uv run ruff check --fix "$file_path" 2>/dev/null
elif command -v ruff &>/dev/null; then
  ruff format "$file_path" 2>/dev/null
  ruff check --fix "$file_path" 2>/dev/null
fi

exit 0
