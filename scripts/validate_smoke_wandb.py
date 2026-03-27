"""Validate wandb metrics from smoke test pipeline runs."""

import argparse
import math
import sys

import wandb


def get_run_by_name(project: str, name: str) -> wandb.apis.public.Run:
    """Find the most recent run matching the given name."""
    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": name}, per_page=1)
    runs = list(runs)
    if not runs:
        print(f"FAIL: No run found with name '{name}' in project '{project}'")
        sys.exit(1)
    return runs[0]


def assert_metric_exists(history: list[dict], key: str, run_name: str):
    """Assert at least one row has a non-None, finite value for key."""
    values = [r[key] for r in history if r.get(key) is not None]
    if not values:
        print(f"  FAIL: '{key}' never logged in run '{run_name}'")
        return False
    for v in values:
        if not math.isfinite(v):
            print(f"  FAIL: '{key}' has non-finite value {v} in run '{run_name}'")
            return False
    print(f"  OK: '{key}' logged {len(values)} times, all finite")
    return True


def validate_run(project: str, name: str, expected_training: bool, expected_eval: bool) -> bool:
    """Validate a single wandb run."""
    print(f"\nValidating run: {name}")

    run = get_run_by_name(project, name)
    print(f"  Run ID: {run.id}, State: {run.state}")

    if run.state != "finished":
        print(f"  FAIL: Run state is '{run.state}', expected 'finished'")
        return False
    print(f"  OK: Run finished successfully")

    history = run.history(pandas=False)
    print(f"  History rows: {len(history)}")

    passed = True

    # Training metrics
    if expected_training:
        for key in ["wm_loss", "actor_loss", "critic_loss", "recon_loss", "kl_loss"]:
            if not assert_metric_exists(history, key, name):
                passed = False

    # Episode metrics
    if expected_training:
        for key in ["episode_reward", "episode_count"]:
            if not assert_metric_exists(history, key, name):
                passed = False

    # Eval metrics
    if expected_eval:
        for key in ["eval/reward", "eval/success", "eval/spl", "eval/episode_length"]:
            if not assert_metric_exists(history, key, name):
                passed = False

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate smoke test wandb runs")
    parser.add_argument("--project", required=True, help="wandb project path")
    parser.add_argument("--run_name", required=True, help="Name of the training run")
    parser.add_argument("--resume_name", required=True, help="Name of the resume run")
    args = parser.parse_args()

    all_passed = True

    # Validate training run (should have training + eval metrics)
    if not validate_run(args.project, args.run_name,
                        expected_training=True, expected_eval=True):
        all_passed = False

    # Validate resume run (should have training + eval metrics)
    if not validate_run(args.project, args.resume_name,
                        expected_training=True, expected_eval=True):
        all_passed = False

    print()
    if all_passed:
        print("ALL ASSERTIONS PASSED")
        sys.exit(0)
    else:
        print("SOME ASSERTIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
