#!/usr/bin/env python3
"""Duell-3 scoring: read a long-format metrics.csv, score against C per seed."""
import csv
import sys

REFS = {
    "42": {"hits": 1, "softspl": 0.0866, "dtg": 6.379, "spl": 0.0119,
           "ms": 66.8, "episodes": 44},
    "43": {"hits": 1, "softspl": 0.0539, "dtg": 4.975, "spl": 0.0062,
           "ms": 69.1, "episodes": 41},
}
WEIGHTS = {"hits": 0.45, "softspl": 0.15, "dtg": 0.15, "spl": 0.10,
           "ms": 0.10, "episodes": 0.05}
HIGHER = {"hits": True, "softspl": True, "dtg": False, "spl": True,
          "ms": False, "episodes": True}
CAPS = {"hits": 2.0, "softspl": 1.0, "dtg": 1.0, "spl": 1.0, "ms": 1.0,
        "episodes": 1.0}


def read_metrics(path):
    last = {}
    hits = 0
    max_step = 0
    with open(path) as fh:
        for row in csv.DictReader(fh):
            metric = row["metric"]
            try:
                value = float(row["value"])
            except ValueError:
                continue
            step = int(float(row["step"]))
            max_step = max(max_step, step)
            last[metric] = value
            if metric == "episode/success" and value == 1:
                hits += 1
    return last, hits, max_step


def main(path, seed):
    last, hits, max_step = read_metrics(path)
    vals = {
        "hits": hits,
        "softspl": last.get("metrics/softspl"),
        "dtg": last.get("metrics/dtg"),
        "spl": last.get("metrics/spl"),
        "ms": last.get("perf/ms_per_step_interval"),
        "episodes": last.get("episode/count"),
    }
    ref = REFS[seed]
    score = 0.0
    print(f"{path} seed={seed} N={max_step} sr={last.get('metrics/sr')}")
    for key, val in vals.items():
        if val is None:
            print(f"  {key:9s} MISSING")
            continue
        r = ref[key]
        rel = (val - r) / abs(r) if HIGHER[key] else (r - val) / abs(r)
        rel = max(-CAPS[key], min(CAPS[key], rel))
        contrib = WEIGHTS[key] * rel
        score += contrib
        print(f"  {key:9s} val={val:10.4f} ref={r:10.4f} rel={rel:+.4f} "
              f"contrib={contrib:+.4f}")
    print(f"  SCORE = {score:+.4f}")
    return score


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "42")
