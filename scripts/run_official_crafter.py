"""Run official DreamerV3 on Crafter in isolated venv, output metrics to CSV.

This script:
1. Sets up a venv with the official repo's dependencies (if needed)
2. Runs the official dreamerv3 training as a subprocess
3. Parses the resulting metrics.jsonl into our CSV format
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OFFICIAL_DIR = os.path.join(REPO_ROOT, "external", "dreamerv3-official")
VENV_DIR = os.path.join(REPO_ROOT, "external", "dreamerv3-official-venv")


def find_python3():
    """Find a Python >=3.10 interpreter for the venv (JAX 0.4.33 requires it)."""
    import shutil
    for name in ("python3.12", "python3.11", "python3.10"):
        path = shutil.which(name)
        if path:
            return path
    return sys.executable  # fallback


def setup_venv():
    """Create venv and install official dreamerv3 dependencies."""
    python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.exists(python):
        print(f"Venv already exists at {VENV_DIR}")
        return python

    base_python = find_python3()
    print(f"Creating venv at {VENV_DIR} with {base_python}...")
    subprocess.run([base_python, "-m", "venv", VENV_DIR], check=True)

    pip = os.path.join(VENV_DIR, "bin", "pip")
    subprocess.run([pip, "install", "--upgrade", "pip"], check=True)

    # Install core deps (skip Atari/ALE which are not needed for Crafter)
    print("Installing dependencies...")
    subprocess.run([pip, "install",
        "jax[cuda12]==0.4.33",
        "chex", "optax", "einops", "jaxtyping", "tqdm",
        "colored_traceback", "ipdb",
        "elements>=3.19.1", "ninjax>=3.5.1", "portal>=3.5.0",
        "scope>=0.4.4", "granular>=0.20.3",
        "numpy<2", "google-resumable-media>=2.7.2",
        "crafter", "ruamel.yaml",
    ], check=True)

    print("Venv setup complete.")
    return python


def run_official(python, logdir, steps, seed, log_every_secs):
    """Run official dreamerv3 training on GPU with size12m model."""
    cmd = [
        python, "-m", "dreamerv3.main",
        "--configs", "crafter", "size12m",
        "--logdir", logdir,
        "--run.steps", str(steps),
        "--run.envs", "1",
        "--run.log_every", str(log_every_secs),
        "--run.save_every", "9999999",  # don't save checkpoints
        "--run.report_every", "9999999",  # skip report for speed
        "--run.debug", "True",  # single-threaded for reproducibility
        "--jax.platform", "cuda",
        "--seed", str(seed),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=OFFICIAL_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Official dreamerv3 exited with code {result.returncode}")
    return result.returncode


def parse_jsonl_to_csv(logdir, csv_path):
    """Convert official metrics.jsonl to our CSV format."""
    jsonl_path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"ERROR: {jsonl_path} not found")
        # Try to find any jsonl files
        for f in os.listdir(logdir):
            if f.endswith(".jsonl"):
                print(f"  Found: {os.path.join(logdir, f)}")
        return False

    # Metrics we want to extract (official names → our names)
    metric_map = {
        "train/loss/dyn": "loss/dyn",
        "train/loss/rep": "loss/enc",
        "train/loss/image": "loss/image",
        "train/loss/rew": "loss/rew",
        "train/loss/con": "loss/con",
        "train/loss/policy": "loss/policy",
        "train/loss/value": "loss/value",
        "train/loss/repval": "loss/repval",
        "episode/score": "episode/score",
        "episode/length": "episode/length",
        "train/rew": "imag_reward",
        "train/ret": "imag_return",
        "train/entropy": "entropy",
        "fps/train": "fps/train",
        "fps/policy": "fps/policy",
    }

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            step = data.get("step", 0)
            for official_key, our_key in metric_map.items():
                if official_key in data:
                    val = data[official_key]
                    if isinstance(val, (int, float)):
                        rows.append([step, our_key, val])
            # Also capture any train/loss/* we might have missed
            for k, v in data.items():
                if k.startswith("train/loss/") and isinstance(v, (int, float)):
                    mapped = k.replace("train/", "")
                    if not any(r[1] == mapped and r[0] == step for r in rows):
                        rows.append([step, mapped, v])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])
        writer.writerows(rows)

    print(f"Parsed {len(rows)} metric entries to {csv_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every_secs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default=None,
                        help="Override logdir for official run (default: temp dir)")
    parser.add_argument("--skip_run", action="store_true",
                        help="Skip training, just parse existing logdir")
    args = parser.parse_args()

    if args.logdir:
        logdir = args.logdir
    else:
        logdir = os.path.join(REPO_ROOT, "output", "official_crafter_run")

    os.makedirs(logdir, exist_ok=True)

    if not args.skip_run:
        python = setup_venv()
        run_official(python, logdir, args.steps, args.seed, args.log_every_secs)

    parse_jsonl_to_csv(logdir, args.output)


if __name__ == "__main__":
    main()
