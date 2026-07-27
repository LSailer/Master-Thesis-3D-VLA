"""Duell-2 (r2) scoring for a single metrics.csv (long format: step,metric,value).

Reading protocol (binding, from GOAL.md):
- Treffer   = count of rows with metric == "episode/success" and value == 1
- softspl   = last logged value of metrics/softspl (last = largest step)
- dtg       = last value of metrics/dtg
- spl       = last value of metrics/spl
- Episoden  = last value of episode/count
- ms/Step   = last value of perf/ms_per_step_interval (noted if missing)
- N         = largest step in the file

Note: metrics.csv is NOT step-sorted; "last" is resolved by max step.

Usage: python score.py path/to/metrics.csv [more.csv ...]
"""

import csv
import sys

REF = {
    "treffer": 1.0,
    "softspl": 0.0605,
    "dtg": 5.193,
    "spl": 0.0201,
    "ms_per_step": 134.1,
    "episoden": 18.0,
}

# (key, weight, direction) - "hoch" = higher is better, "niedrig" = lower is better
WEIGHTS = [
    ("treffer", 0.45, "hoch"),
    ("softspl", 0.15, "hoch"),
    ("dtg", 0.15, "niedrig"),
    ("spl", 0.10, "hoch"),
    ("ms_per_step", 0.10, "niedrig"),
    ("episoden", 0.05, "hoch"),
]


# Unscored diagnostics, read off as last value and reported alongside the score.
EXTRA_METRICS = ["action/forward_pct", "episode/path_length"]


def read_metrics(path):
    """Parse a long-format metrics.csv.

    Args:
        path: Path to the CSV file with columns step,metric,value.

    Returns:
        Dict with the raw read-off values per the protocol above. ms_per_step
        is None when perf/ms_per_step_interval never appears in the file.
    """
    treffer = 0
    last = {}  # metric -> (step, value), keeping the largest step
    max_step = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            metric = row["metric"]
            max_step = max(max_step, step)
            try:
                value = float(row["value"])
            except ValueError:
                # Non-numeric metric (e.g. episode/goal = "chair"); step still counts for N.
                continue
            if metric == "episode/success" and value == 1:
                treffer += 1
            prev = last.get(metric)
            if prev is None or step >= prev[0]:
                last[metric] = (step, value)

    def last_val(name):
        entry = last.get(name)
        return entry[1] if entry is not None else None

    return {
        "treffer": float(treffer),
        "softspl": last_val("metrics/softspl"),
        "dtg": last_val("metrics/dtg"),
        "spl": last_val("metrics/spl"),
        "episoden": last_val("episode/count"),
        "ms_per_step": last_val("perf/ms_per_step_interval"),
        "n": max_step,
        "extras": {name: last_val(name) for name in EXTRA_METRICS},
    }


def rel(value, ref, direction):
    """Relative delta vs. reference; positive = better."""
    if direction == "hoch":
        return (value - ref) / abs(ref)
    return (ref - value) / abs(ref)


def score(values):
    """Compute capped, weighted score contributions.

    Args:
        values: Dict from read_metrics.

    Returns:
        (total, contribs) where contribs maps key -> (raw_rel, capped_rel,
        weighted_contribution) or None when the value is missing.
    """
    total = 0.0
    contribs = {}
    for key, weight, direction in WEIGHTS:
        value = values[key]
        if value is None:
            contribs[key] = None
            continue
        raw = rel(value, REF[key], direction)
        cap_hi = 2.0 if key == "treffer" else 1.0
        capped = max(-1.0, min(cap_hi, raw))
        contrib = weight * capped
        contribs[key] = (raw, capped, contrib)
        total += contrib
    return total, contribs


def report(path):
    values = read_metrics(path)
    total, contribs = score(values)
    print(f"=== {path}")
    fmt = {
        "treffer": ".0f",
        "softspl": ".4f",
        "dtg": ".3f",
        "spl": ".4f",
        "ms_per_step": ".1f",
        "episoden": ".0f",
    }
    for key, weight, _direction in WEIGHTS:
        value = values[key]
        if value is None:
            print(f"  {key:<12} MISSING (weight {weight:.2f} contributes 0.0 - vermerkt)")
            continue
        raw, capped, contrib = contribs[key]
        capnote = "" if raw == capped else f" (capped from {raw:+.4f})"
        print(
            f"  {key:<12} {value:{fmt[key]}}  rel={capped:+.4f}{capnote}"
            f"  w={weight:.2f}  contrib={contrib:+.4f}"
        )
    print(f"  N (max step) {values['n']}")
    for name, val in values["extras"].items():
        shown = "MISSING" if val is None else f"{val:.4f}"
        print(f"  {name:<24} {shown} (unscored)")
    print(f"  SCORE        {total:+.4f}")
    return values, total, contribs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python score.py metrics.csv [more.csv ...]")
    for p in sys.argv[1:]:
        report(p)
