"""Run PyTorch r2dreamer on Crafter, output metrics to CSV.

This script:
1. Runs external/r2dreamer/train.py via subprocess with Hydra config overrides
2. Supports rep_loss=dreamer (DreamerV3 with decoder) and rep_loss=r2dreamer
3. Parses TensorBoard events or metrics.jsonl from the logdir into CSV format
4. Outputs CSV with columns: step, metric, value
"""

import argparse
import csv
import json
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R2DREAMER_DIR = os.path.join(REPO_ROOT, "external", "r2dreamer")


def run_r2dreamer(logdir, steps, seed, rep_loss="r2dreamer"):
    """Run r2dreamer training on Crafter with the given rep_loss mode."""
    cmd = [
        sys.executable, "train.py",
        "env=crafter",
        "model=size12M",
        f"model.rep_loss={rep_loss}",
        "model.compile=False",
        f"seed={seed}",
        f"logdir={logdir}",
        f"trainer.steps={steps}",
        # Disable Hydra output dir redirection so logdir is used directly
        f"hydra.run.dir={logdir}",
    ]
    print(f"Running ({rep_loss}): {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=R2DREAMER_DIR)
    if result.returncode != 0:
        print(f"WARNING: r2dreamer ({rep_loss}) exited with code {result.returncode}")
    return result.returncode


def parse_tensorboard_to_rows(logdir):
    """Parse TensorBoard event files and return list of [step, metric, value] rows."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard package not available, skipping TensorBoard parsing.")
        return []

    rows = []
    # Search logdir recursively for event files
    for root, _dirs, files in os.walk(logdir):
        for fname in files:
            if not fname.startswith("events.out.tfevents"):
                continue
            event_path = os.path.join(root, fname)
            try:
                ea = EventAccumulator(event_path)
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    for scalar_event in ea.Scalars(tag):
                        rows.append([scalar_event.step, tag, scalar_event.value])
            except Exception as e:
                print(f"WARNING: could not parse {event_path}: {e}")

    return rows


def parse_jsonl_to_rows(logdir):
    """Parse any .jsonl files in logdir and return list of [step, metric, value] rows."""
    rows = []
    for root, _dirs, files in os.walk(logdir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            jsonl_path = os.path.join(root, fname)
            try:
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        step = data.get("step", 0)
                        for k, v in data.items():
                            if k == "step":
                                continue
                            if isinstance(v, (int, float)):
                                rows.append([step, k, v])
            except Exception as e:
                print(f"WARNING: could not parse {jsonl_path}: {e}")

    return rows


def write_csv(rows, csv_path):
    """Write rows to CSV file with columns: step, metric, value."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metric entries to {csv_path}")


def collect_and_write_metrics(logdir, csv_path):
    """Try TensorBoard first, fall back to jsonl, write CSV."""
    rows = parse_tensorboard_to_rows(logdir)
    if rows:
        print(f"Parsed {len(rows)} rows from TensorBoard events.")
    else:
        print("No TensorBoard data found, trying .jsonl files...")
        rows = parse_jsonl_to_rows(logdir)
        if rows:
            print(f"Parsed {len(rows)} rows from .jsonl files.")
        else:
            print(f"WARNING: no metrics found in {logdir}")

    if rows:
        write_csv(rows, csv_path)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run PyTorch r2dreamer on Crafter and export metrics to CSV."
    )
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total training steps (default: 100000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "output", "r2dreamer_crafter"),
                        help="Directory to store run logs and CSV output")
    parser.add_argument("--rep_loss", choices=["dreamer", "r2dreamer", "both"],
                        default="both",
                        help="Representation loss mode: dreamer, r2dreamer, or both (default: both)")
    parser.add_argument("--skip_run", action="store_true",
                        help="Skip training, just parse existing logdir(s)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    modes = ["dreamer", "r2dreamer"] if args.rep_loss == "both" else [args.rep_loss]

    for mode in modes:
        logdir = os.path.join(args.output_dir, f"{mode}_seed{args.seed}")
        csv_path = os.path.join(args.output_dir, f"{mode}_seed{args.seed}_metrics.csv")
        os.makedirs(logdir, exist_ok=True)

        if not args.skip_run:
            run_r2dreamer(logdir, args.steps, args.seed, rep_loss=mode)

        collect_and_write_metrics(logdir, csv_path)


if __name__ == "__main__":
    main()
