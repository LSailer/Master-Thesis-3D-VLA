#!/usr/bin/env bash
set -euo pipefail

if ! command -v semgrep >/dev/null 2>&1; then
  echo "error: semgrep is not installed or not on PATH" >&2
  echo "Install Semgrep or run this through the GitHub policy workflow." >&2
  exit 127
fi

semgrep ci \
  --code \
  --include='src/**' \
  --include='scripts/**'
