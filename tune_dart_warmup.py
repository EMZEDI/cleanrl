"""
Analyze DART warmup_frac tuning results from TensorBoard logs.

After the SLURM job finishes, run this from the project root:
    python tune_dart_warmup.py [--runs-dir <path>]

It scans all TensorBoard event files written by the tuning job,
extracts the last-200-step average of `charts/episodic_return`
per (warmup_frac, env) pair, and reports the best warmup_frac
on average across all 50 dm_control environments.
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


METRIC = "charts/episodic_return"
LAST_N = 200


def extract_metric(run_dir: str, metric: str = METRIC, last_n: int = LAST_N) -> float | None:
    """Read TensorBoard events and return the mean of the last `last_n` scalars."""
    try:
        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if metric not in tags:
            return None
        events = ea.Scalars(metric)
        if len(events) < last_n:
            values = [e.value for e in events]
        else:
            values = [e.value for e in events[-last_n:]]
        return float(np.mean(values)) if values else None
    except Exception as e:
        print(f"  Warning: could not read {run_dir}: {e}")
        return None


def parse_run_name(run_name: str):
    """
    Run names follow the pattern:
      dm_control/<domain>-<task>-v0__dart_warmup_<frac>__<seed>__<timestamp>
    Returns (env_id, warmup_frac) or None.
    """
    # e.g. "dm_control/cheetah-run-v0__dart_warmup_0.400__42__1234567890"
    m = re.match(r"^(.+?)__dart_warmup_([\d.]+)__(\d+)__(\d+)$", run_name)
    if m:
        env_id = m.group(1)
        warmup_frac = float(m.group(2))
        return env_id, warmup_frac
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze DART warmup_frac tuning")
    parser.add_argument("--runs-dir", type=str, default="runs",
                        help="Directory containing TensorBoard run folders")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: runs directory not found: {runs_dir}")
        return

    # Collect results: {warmup_frac: {env_id: avg_return}}
    results = defaultdict(dict)

    # Run names contain / in env_id, e.g.:
    #   runs/dm_control/cheetah-run-v0__dart_warmup_0.350__42__1234567890
    # So actual run dirs are 2 levels deep: runs/<domain>/<rest>
    # We glob for event files and reconstruct.
    run_candidates = []
    for domain_dir in sorted(runs_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        for sub in sorted(domain_dir.iterdir()):
            if sub.is_dir():
                # Reconstruct full run name as "domain/sub"
                run_candidates.append((f"{domain_dir.name}/{sub.name}", str(sub)))

    total = len(run_candidates)
    print(f"Scanning {total} run directories in {runs_dir} ...")

    for i, (run_name, run_path) in enumerate(run_candidates):
        parsed = parse_run_name(run_name)
        if parsed is None:
            continue
        env_id, warmup_frac = parsed
        val = extract_metric(run_path)
        if val is not None:
            results[warmup_frac][env_id] = val
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total} directories")

    if not results:
        print("ERROR: No valid tuning runs found. Check that runs are in the expected directory.")
        return

    # Compute average across all envs for each warmup_frac
    print("\n" + "=" * 70)
    print("DART warmup_frac Tuning Results")
    print("=" * 70)
    print(f"{'warmup_frac':>14s}  {'#envs':>6s}  {'avg_return':>12s}")
    print("-" * 36)

    summary = {}
    for wf in sorted(results.keys()):
        env_scores = results[wf]
        avg = np.mean(list(env_scores.values()))
        summary[wf] = (avg, len(env_scores))
        print(f"{wf:>14.3f}  {len(env_scores):>6d}  {avg:>12.2f}")

    # Best
    best_wf = max(summary, key=lambda k: summary[k][0])
    best_avg, best_n = summary[best_wf]

    print("-" * 36)
    print(f"\n>>> BEST warmup_frac = {best_wf:.3f}  "
          f"(avg return = {best_avg:.2f} across {best_n} envs)")
    print()

    # Also print per-env breakdown for best
    print(f"Per-env scores for best warmup_frac={best_wf:.3f}:")
    print(f"{'env_id':<50s}  {'return':>10s}")
    print("-" * 62)
    for env_id in sorted(results[best_wf].keys()):
        print(f"{env_id:<50s}  {results[best_wf][env_id]:>10.2f}")


if __name__ == "__main__":
    main()
