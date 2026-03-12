#!/usr/bin/env bash
# PreToolUse hook: block dangerous git commands
# Exit 0 = allow, Exit 2 = block

input=$(cat)
tool_name=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null)

[[ "$tool_name" != "Bash" ]] && exit 0

command=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

# Only check git commands
[[ "$command" != *"git "* ]] && exit 0

# Block dangerous patterns
dangerous_patterns=(
  "push --force"
  "push -f "
  "push -f$"
  "push --force-with-lease"
  "reset --hard"
  "clean -f"
  "clean -fd"
  "branch -D"
  "checkout \."
  "checkout -- \."
  "restore \."
)

for pattern in "${dangerous_patterns[@]}"; do
  if echo "$command" | grep -qE "git\s+.*${pattern}"; then
    echo "BLOCKED: dangerous git command detected: $command" >&2
    exit 2
  fi
done

exit 0
