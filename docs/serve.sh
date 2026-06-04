#!/usr/bin/env bash
# Serve this docs/ folder over HTTP so the reports and Plotly viewers can be
# opened in a browser (e.g. via an SSH tunnel or cmux port-forward).
#
# Usage:
#   ./docs/serve.sh            # serve docs/ on port 3000
#   ./docs/serve.sh 8080       # serve docs/ on a custom port
#
# Then browse http://localhost:<port>/ (after forwarding the port to your
# local machine). Press Ctrl+C to stop.

set -euo pipefail

# Anchor to this script's directory so docs/ is the document root regardless
# of the current working directory. Relative asset paths (images/...) resolve.
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-3000}"

# Fail early with a clear message instead of Python's terse "Address already
# in use" traceback.
if ss -ltn 2>/dev/null | grep -qE "[:.]${PORT}\b"; then
  echo "Port ${PORT} is already in use. Pick another: ./docs/serve.sh <port>" >&2
  exit 1
fi

echo "Serving ${DOCS_DIR} at http://localhost:${PORT}/  (Ctrl+C to stop)"
exec python3 -m http.server "${PORT}" --bind 0.0.0.0 --directory "${DOCS_DIR}"
