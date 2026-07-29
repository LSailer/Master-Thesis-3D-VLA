"""Per-rule count ratchet over linter/typechecker output.

Reads diagnostics from stdin, counts them per rule id (pylint message ids
like C0103, basedpyright rules like reportExplicitAny), and compares against
the committed baseline. A rule whose count exceeds the baseline fails the
gate and every finding of that rule is printed, so the offending lines are
actionable feedback. When the run is at or below baseline everywhere and
strictly below somewhere, the baseline file is rewritten with the lower
counts: the ratchet only ever tightens.

Counts, not positions: adding a finding while removing another of the same
rule is invisible to the ratchet. That is the accepted trade for a baseline
file that stays human-readable and identical in shape for both tools.

Usage:
    pylint src -f json --exit-zero | python scripts/gate/ratchet.py \
        --format pylint --baseline scripts/gate/pylint-baseline.json
    basedpyright --outputjson | python scripts/gate/ratchet.py \
        --format basedpyright --baseline scripts/gate/basedpyright-baseline.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    """One diagnostic in tool-independent form."""

    rule: str
    path: str
    line: int
    message: str


def _parse_pylint(payload: object) -> list[Finding]:
    """Convert ``pylint -f json`` output (a list of message dicts)."""
    assert isinstance(payload, list)
    return [
        Finding(
            rule=str(msg["message-id"]),
            path=str(msg["path"]),
            line=int(msg["line"]),
            message=f"{msg['symbol']}: {msg['message']}",
        )
        for msg in payload
    ]


def _parse_basedpyright(payload: object) -> list[Finding]:
    """Convert ``basedpyright --outputjson`` output (generalDiagnostics)."""
    assert isinstance(payload, dict)
    return [
        Finding(
            rule=str(diag.get("rule", diag["severity"])),
            path=str(diag["file"]),
            line=int(diag["range"]["start"]["line"]) + 1,
            message=str(diag["message"]).split("\n", maxsplit=1)[0],
        )
        for diag in payload["generalDiagnostics"]
    ]


_PARSERS = {"pylint": _parse_pylint, "basedpyright": _parse_basedpyright}


def _load_baseline(path: Path) -> dict[str, int]:
    """Return the committed per-rule counts.

    Args:
        path: Baseline JSON file mapping rule id to allowed count.

    Returns:
        The baseline mapping; empty when the file does not exist yet.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    counts = data.get("counts", data)
    return {str(key): int(value) for key, value in counts.items()}


def main() -> int:
    """Compare stdin diagnostics against the baseline.

    Returns:
        0 when every rule is at or below its baseline count, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--format", choices=sorted(_PARSERS), required=True)
    args = parser.parse_args()

    findings = _PARSERS[args.format](json.load(sys.stdin))
    counts = Counter(finding.rule for finding in findings)
    baseline = _load_baseline(args.baseline)

    violations = {
        rule: count
        for rule, count in sorted(counts.items())
        if count > baseline.get(rule, 0)
    }
    if violations:
        print(f"{args.format.upper()} RATCHET FAILED - counts above baseline:")
        for rule, count in violations.items():
            allowed = baseline.get(rule, 0)
            print(f"\n{rule}: {count} found, {allowed} allowed. All sites:")
            for finding in findings:
                if finding.rule == rule:
                    print(f"  {finding.path}:{finding.line}: {finding.message}")
        print(
            "\nFix the new findings (or reduce old ones of the same rule); "
            "the baseline never moves up."
        )
        return 1

    tightened = {
        rule: allowed
        for rule, allowed in baseline.items()
        if counts.get(rule, 0) < allowed
    }
    if tightened:
        new_counts = {
            rule: count for rule, count in sorted(counts.items()) if count > 0
        }
        args.baseline.write_text(
            json.dumps({"counts": new_counts}, indent=2) + "\n"
        )
        for rule, allowed in sorted(tightened.items()):
            print(f"ratchet tightened: {rule} {allowed} -> {counts.get(rule, 0)}")

    total = sum(counts.values())
    print(f"{args.format} ratchet OK: {total} findings, all within baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
