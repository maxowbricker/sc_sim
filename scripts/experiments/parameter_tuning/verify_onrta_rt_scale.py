#!/usr/bin/env python3
"""
Parameter Verification — ONRTA-RT UTILITY_SCALE

ONRTA-RT draws a random threshold theta from {e^0, ..., e^(lambda-1)}, where:
    lambda = ceil(ln(UTILITY_SCALE + 1))

Scaled pair utility is:  score(w, t) = UTILITY_SCALE / (1 + d_pick)

For the threshold set to be meaningful, the thresholds must span the actual
range of pair scores on the dataset:

  - If the highest threshold e^(lambda-1) exceeds all scores → upper thresholds
    never fire; the policy always falls back to greedy (degenerates).
  - If even the lowest threshold e^0 = 1 exceeds all scores → ALL events
    take the greedy fallback path; randomised policy never activates.
  - Ideal: the highest threshold is exceeded by ~P10–P20 of feasible pairs,
    so the policy meaningfully differentiates acceptance behaviour.

This script loads each dataset, computes pickup-distance statistics from a
random sample of (task, worker) proximity pairs, and reports:
    1. The threshold set produced by the current UTILITY_SCALE
    2. The fraction of sampled pairs above each threshold
    3. A pass/fail verdict and suggested alternative UTILITY_SCALE if needed

No simulations are run — this is a fast, data-only diagnostic.

Usage:
    python scripts/experiments/parameter_tuning/verify_onrta_rt_scale.py
    python scripts/experiments/parameter_tuning/verify_onrta_rt_scale.py --didi-only
    python scripts/experiments/parameter_tuning/verify_onrta_rt_scale.py --scale 50
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, PROJECT_ROOT)

from data.loader import load_workers_tasks
from simulator.spatial_index import fast_manhattan_km

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DIDI_ROOT   = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
DIDI_DAY    = "496528674@qq.com_20161109"

GOWALLA_ROOT   = os.path.join(PROJECT_ROOT, "data", "gowalla")
GOWALLA_KWARGS = dict(
    region                 = "austin",
    date_start             = "2010-09-01",
    date_end               = "2010-09-30",
    task_mode              = "checkin",
    task_window_hours      = 0.5,
    shift_hours            = 8.0,
    dropoff_noise_km       = 2.0,
    compress_to_day        = True,
    workers_per_task_ratio = 0.20,
    random_state           = 42,
)

# Current hardcoded value in onrta_rt.py
DEFAULT_SCALE = 100.0

# How many (task, worker) pairs to sample for distance analysis
N_SAMPLE_TASKS   = 2_000
N_WORKERS_PER_TASK = 10    # nearest workers per sampled task


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_distance_sample(workers, tasks, n_tasks: int,
                             n_workers_per_task: int,
                             seed: int = 42) -> List[float]:
    """Sample pickup distances from n_tasks random tasks × their nearest workers."""
    rng = random.Random(seed)
    task_sample = rng.sample(tasks, min(n_tasks, len(tasks)))
    worker_list = list(workers)

    distances: List[float] = []
    for task in task_sample:
        # Sort all workers by distance to this task's pickup (or sample a subset)
        subset = rng.sample(worker_list, min(n_workers_per_task * 10, len(worker_list)))
        dists = sorted(
            fast_manhattan_km(w.start_lat, w.start_lon, task.pickup_lat, task.pickup_lon)
            for w in subset
        )
        distances.extend(dists[:n_workers_per_task])

    return distances


def threshold_set(scale: float) -> List[float]:
    """Return the ONRTA-RT threshold set for a given UTILITY_SCALE."""
    lam = max(1, math.ceil(math.log(scale + 1)))
    return [math.exp(e) for e in range(lam)]


def analyse(dataset_name: str, distances: List[float], scale: float) -> bool:
    """Print analysis and return True if UTILITY_SCALE passes."""
    scores = [scale / (1.0 + d) for d in distances]
    thresholds = threshold_set(scale)
    lam = len(thresholds)

    pcts   = [5, 10, 25, 50, 75, 90, 95]
    pvalues = np.percentile(distances, pcts)
    svalues = [scale / (1.0 + p) for p in pvalues]

    print(f"\n{'─' * 65}")
    print(f"  {dataset_name}  (UTILITY_SCALE = {scale:.0f})")
    print(f"{'─' * 65}")

    print(f"\n  Pickup-distance percentiles across {len(distances):,} sampled pairs:")
    print(f"  {'Percentile':<12} {'Dist (km)':>10} {'Score':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for p, d, s in zip(pcts, pvalues, svalues):
        print(f"  P{p:<11} {d:>10.2f} {s:>10.2f}")

    print(f"\n  Threshold set  (lambda = {lam}):")
    print(f"  {'Threshold (theta)':>20} {'% pairs >= theta':>18} {'Assessment':>14}")
    print(f"  {'-'*20} {'-'*18} {'-'*14}")

    coverage_flags = []
    for theta in thresholds:
        frac = sum(1 for s in scores if s >= theta) / len(scores)
        pct  = frac * 100.0
        # Good: at least a few percent but not trivially all
        if pct >= 80:
            flag = "easy (all pass)"
        elif 5 <= pct < 80:
            flag = "OK"
        else:
            flag = "VERY RESTRICTIVE"
        coverage_flags.append(pct)
        print(f"  {theta:>20.3f} {pct:>17.1f}% {flag:>14}")

    # Verdict logic
    top_coverage = coverage_flags[-1]   # highest threshold
    bot_coverage = coverage_flags[0]    # lowest threshold (theta=1)

    print()
    passed = True

    if bot_coverage < 10:
        print(f"  ✗ Even the lowest threshold (theta=1) is exceeded by only "
              f"{bot_coverage:.1f}% of pairs.")
        print(f"    The policy almost always falls back to greedy. "
              f"UTILITY_SCALE={scale:.0f} is too HIGH.")
        passed = False
    elif top_coverage > 50:
        print(f"  ✗ The highest threshold (theta={thresholds[-1]:.1f}) is exceeded by "
              f"{top_coverage:.1f}% of pairs.")
        print(f"    Upper thresholds are too permissive — little randomisation effect. "
              f"UTILITY_SCALE={scale:.0f} may be too LOW.")
        passed = False
    elif lam < 3:
        print(f"  ✗ lambda={lam} gives only {lam} threshold level(s) — "
              f"insufficient diversity. Increase UTILITY_SCALE.")
        passed = False
    else:
        print(f"  ✓ UTILITY_SCALE={scale:.0f} looks appropriate for {dataset_name}.")
        print(f"    lambda={lam} levels; highest threshold accepted by "
              f"{top_coverage:.1f}% of pairs (target: 5–30%).")

    # Suggest an alternative if needed
    if not passed:
        # Aim for the top threshold to cover ~P15 of pairs
        target_score = float(np.percentile(scores, 85))  # 85th percentile of scores
        # We want e^(lam-1) ~ target_score → lam-1 = ln(target_score)
        # UTILITY_SCALE = e^(lambda-1+1) - 1 → back out from desired lambda
        suggested_lam = max(3, round(math.log(max(target_score, math.e)) + 1))
        suggested_scale = round(math.exp(suggested_lam) - 1)
        suggested_scale = max(10, min(suggested_scale, 10_000))
        thresholds_new = threshold_set(float(suggested_scale))
        new_top_cov = sum(1 for s in scores if s >= thresholds_new[-1]) / len(scores) * 100
        print(f"\n  Suggested UTILITY_SCALE = {suggested_scale}  "
              f"(lambda={len(thresholds_new)}, "
              f"top threshold coverage ≈ {new_top_cov:.1f}%)")
        print(f"  Update UTILITY_SCALE in simulator/strategies/onrta_rt.py if needed.")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--didi-only",    action="store_true")
    parser.add_argument("--gowalla-only", action="store_true")
    parser.add_argument("--scale",        type=float, default=DEFAULT_SCALE,
                        help=f"UTILITY_SCALE to test (default: {DEFAULT_SCALE})")
    parser.add_argument("--n-tasks",      type=int,   default=N_SAMPLE_TASKS,
                        help="Tasks to sample for distance analysis")
    args = parser.parse_args()

    run_didi    = not args.gowalla_only
    run_gowalla = not args.didi_only
    scale       = args.scale

    print("=" * 65)
    print("  ONRTA-RT  UTILITY_SCALE Verification")
    print(f"  Testing UTILITY_SCALE = {scale:.0f}")
    print(f"  Threshold set: {[f'{t:.2f}' for t in threshold_set(scale)]}")
    print(f"  lambda = {len(threshold_set(scale))}")
    print(f"  Sampling {args.n_tasks:,} tasks × {N_WORKERS_PER_TASK} nearest workers")
    print("=" * 65)

    results = {}

    # ── Gowalla ──────────────────────────────────────────────────────────────
    if run_gowalla:
        print("\n  Loading Gowalla Austin ...", end="  ", flush=True)
        workers_g, tasks_g = load_workers_tasks(
            "gowalla", root_path=GOWALLA_ROOT, **GOWALLA_KWARGS
        )
        print(f"{len(workers_g):,} workers | {len(tasks_g):,} tasks")
        dists_g = compute_distance_sample(
            workers_g, tasks_g, args.n_tasks, N_WORKERS_PER_TASK
        )
        results["Gowalla"] = analyse("Gowalla Austin (Sep 2010)", dists_g, scale)

    # ── Didi ─────────────────────────────────────────────────────────────────
    if run_didi:
        day_path = os.path.join(DIDI_ROOT, DIDI_DAY)
        print(f"\n  Loading Didi {DIDI_DAY} ...", end="  ", flush=True)
        workers_d, tasks_d = load_workers_tasks("didi", root_path=day_path)
        print(f"{len(workers_d):,} workers | {len(tasks_d):,} tasks")
        dists_d = compute_distance_sample(
            workers_d, tasks_d, args.n_tasks, N_WORKERS_PER_TASK
        )
        results["Didi"] = analyse("Didi Chengdu 20161109", dists_d, scale)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    for ds, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {ds:<20} UTILITY_SCALE={scale:.0f}  →  {status}")

    if all(results.values()):
        print(f"\n  No changes needed — UTILITY_SCALE={scale:.0f} is valid for all datasets.")
    else:
        print(f"\n  Update UTILITY_SCALE in simulator/strategies/onrta_rt.py")
        print(f"  and re-run this script to confirm.")
    print()


if __name__ == "__main__":
    main()
