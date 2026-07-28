"""Per-message-id ratchet over pylint output.

Reads pylint JSON diagnostics from stdin, counts them per message id
(C0103, W0212, ...), and compares against the committed baseline. A message
id whose count exceeds the baseline fails the gate and every message of that
id is printed, so the offending lines are actionable feedback. When the run
is at or below baseline everywhere and strictly below somewhere, the baseline
file is rewritten with the lower counts: the ratchet only ever tightens.

Usage:
    pylint src -f json --exit-zero | python scripts/gate/pylint_ratchet.py \
        --baseline scripts/gate/pylint-baseline.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_baseline(path: Path) -> dict[str, int]:
    """Return the committed per-message-id counts.

    Args:
        path: Baseline JSON file mapping message id to allowed count.

    Returns:
        The baseline mapping; empty when the file does not exist yet.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    counts = data.get("counts", data)
    return {str(key): int(value) for key, value in counts.items()}


def main() -> int:
    """Compare pylint stdin diagnostics against the baseline.

    Returns:
        0 when every message id is at or below its baseline count,
        1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()

    messages = json.load(sys.stdin)
    counts = Counter(str(msg["message-id"]) for msg in messages)
    baseline = _load_baseline(args.baseline)

    violations = {
        msg_id: count
        for msg_id, count in sorted(counts.items())
        if count > baseline.get(msg_id, 0)
    }
    if violations:
        print("PYLINT RATCHET FAILED - counts above the committed baseline:")
        for msg_id, count in violations.items():
            allowed = baseline.get(msg_id, 0)
            print(f"\n{msg_id}: {count} found, {allowed} allowed. All sites:")
            for msg in messages:
                if msg["message-id"] == msg_id:
                    print(
                        f"  {msg['path']}:{msg['line']}: "
                        f"{msg['symbol']}: {msg['message']}"
                    )
        print(
            "\nFix the new findings (or reduce old ones of the same id); "
            "the baseline never moves up."
        )
        return 1

    tightened = {
        msg_id: allowed
        for msg_id, allowed in baseline.items()
        if counts.get(msg_id, 0) < allowed
    }
    if tightened:
        new_counts = {
            msg_id: count for msg_id, count in sorted(counts.items()) if count > 0
        }
        args.baseline.write_text(
            json.dumps({"counts": new_counts}, indent=2) + "\n"
        )
        for msg_id, allowed in sorted(tightened.items()):
            print(
                f"ratchet tightened: {msg_id} "
                f"{allowed} -> {counts.get(msg_id, 0)}"
            )

    total = sum(counts.values())
    print(f"pylint ratchet OK: {total} findings, all within baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
