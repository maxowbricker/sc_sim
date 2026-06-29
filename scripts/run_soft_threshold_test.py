#!/usr/bin/env python3
"""
Soft-threshold sensitivity test — Composite (static) on Didi 20161109.

Runs Composite with the current optimal static weights but varying soft_threshold,
plus a Greedy reference row, to determine whether soft_threshold meaningfully
affects outcomes and whether it should be included in the Pareto sweep.

Fixed weights: fairness_weight=1.0, starvation_weight=0.2, utility_weight=1.0,
               gamma=0.1, k=15

Soft-threshold values tested: 0.0, 0.05 (current), 0.1, 0.2, 0.3, 1.0

Usage:
    python scripts/run_soft_threshold_test.py
    python scripts/run_soft_threshold_test.py --output soft_threshold_results.csv
"""

from __future__ import annotations

import argparse
import copy
import csv as _csv
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

DATA_ROOT  = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY = "496528674@qq.com_20161109"

# Fixed composite weights — the current paper configuration
BASE_WEIGHTS = dict(
    fairness_weight=1.0,
    starvation_weight=0.2,
    utility_weight=1.0,
    gamma=0.1,
    k=15,
)

# Soft-threshold values to test (0.05 is the current default)
SOFT_THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.3, 1.0]

TIMEOUT_SEC = 300


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(stats: Dict[str, Any], workers) -> Dict[str, Any]:
    completed  = stats.get("completed_tasks", 0)
    total      = stats.get("total_tasks", 1)
    wait_times = stats.get("wait_times", [])

    tar      = completed / total if total else 0.0
    jfi      = stats.get("final_jains_fairness_index", 0.0)
    jfi_earn = stats.get("final_jfi_earnings", 0.0)

    worker_task_counts = [w.completed_tasks for w in workers]
    jfi_rate = sum(1 for c in worker_task_counts if c > 0) / max(len(worker_task_counts), 1)

    avg_wait = float(np.mean(wait_times))            if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95))  if wait_times else 0.0

    return {
        "Completed":       completed,
        "Total":           total,
        "TAR":             tar,
        "JFI (tasks)":     jfi,
        "JFI (earnings)":  jfi_earn,
        "JFI rate":        jfi_rate,
        "Avg Wait (m)":    avg_wait,
        "P95 Wait (m)":    p95_wait,
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _sim_thread(sim, exc_holder):
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_one(workers_template, tasks_template, strategy_key: str,
            params: dict, timeout_sec: float) -> Optional[Dict[str, Any]]:
    cfg = create_composite_config(assignment_strategy=strategy_key, **params)
    t0  = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator
        sim = EventSimulator(
            copy.deepcopy(workers_template),
            copy.deepcopy(tasks_template),
            cfg,
        )
        sim.reset()

        thread = threading.Thread(
            target=_sim_thread, args=(sim, exc_holder), daemon=True
        )
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            return None

        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None

    m = extract_metrics(stats, workers)
    m["elapsed_s"] = time.time() - t0
    return m


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

COL_W  = 34
STAT_W = 13

DISPLAY_KEYS = [
    "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
]


def _fmt(v: Any, key: str) -> str:
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def _header() -> str:
    return f"{'Config':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS) + f"  {'Time (s)':>8}"


def _sep() -> str:
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 10)


def _row(name: str, m: Dict[str, Any]) -> str:
    cells = "".join(f" {_fmt(m.get(k, 0), k):>{STAT_W}}" for k in DISPLAY_KEYS)
    return f"{name:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>8.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path (default: soft_threshold_test_20161109.csv in project root)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC,
                        help=f"Per-run timeout in seconds (default: {TIMEOUT_SEC})")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        PROJECT_ROOT, "soft_threshold_test_20161109.csv"
    )

    # Build run list: Greedy reference + one Composite run per threshold
    runs: List[tuple] = [("Greedy (reference)", "greedy", {})]
    for st in SOFT_THRESHOLDS:
        label = f"Composite  st={st:.3f}" + ("  ← current" if st == 0.05 else "")
        runs.append((label, "composite", {**BASE_WEIGHTS, "soft_threshold": st}))

    print("=" * 80)
    print("  Soft-threshold Sensitivity Test — Composite (static)")
    print(f"  Day:       {TARGET_DAY}")
    print(f"  Weights:   fw=1.0  sw=0.2  uw=1.0  gamma=0.1  k=15")
    print(f"  Thresholds: {SOFT_THRESHOLDS}")
    print(f"  Total runs: {len(runs)}")
    print(f"  Est. time:  ~{len(runs) * 85 / 60:.0f} min")
    print(f"  Output:    {output_path}")
    print("=" * 80)

    # Load data once
    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY}...")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_template):,} workers | {len(tasks_template):,} tasks\n")

    results = []

    for label, strategy_key, params in runs:
        print(f"  Running: {label} ...", end="  ", flush=True)
        m = run_one(workers_template, tasks_template, strategy_key, params, args.timeout)

        if m is None:
            print("TIMEOUT / FAILED")
            results.append({"label": label, "soft_threshold": params.get("soft_threshold", "—"), "_failed": True})
            continue

        print(
            f"TAR={m['TAR']:.4f}  "
            f"JFI={m['JFI (tasks)']:.4f}  "
            f"wait={m['Avg Wait (m)']:.2f}m  "
            f"[{m['elapsed_s']:.1f}s]"
        )
        m["label"]          = label
        m["soft_threshold"] = params.get("soft_threshold", "—")
        m["strategy"]       = strategy_key
        m.update({k: params.get(k, "—") for k in
                  ("fairness_weight", "starvation_weight", "utility_weight", "gamma", "k")})
        results.append(m)

    # Print summary table
    successful = [r for r in results if not r.get("_failed")]
    print(f"\n\n{'=' * 80}")
    print(f"  RESULTS — soft_threshold sensitivity (Didi {TARGET_DAY})")
    print(f"  Fixed: fw=1.0  sw=0.2  uw=1.0  gamma=0.1  k=15")
    print(f"{'=' * 80}\n")
    print(_sep())
    print(_header())
    print(_sep())
    for r in successful:
        print(_row(r["label"], r))
    print(_sep())

    # Highlight whether any threshold value changes outcomes
    composites = [r for r in successful if r.get("strategy") == "composite"]
    if len(composites) >= 2:
        jfis  = [r["JFI (tasks)"]   for r in composites]
        waits = [r["Avg Wait (m)"]  for r in composites]
        tars  = [r["TAR"]           for r in composites]
        print(f"\n  Δ JFI   across thresholds: {max(jfis)  - min(jfis):.4f}")
        print(f"  Δ Wait  across thresholds: {max(waits) - min(waits):.3f} m")
        print(f"  Δ TAR   across thresholds: {max(tars)  - min(tars):.4f}")
        print()
        if (max(jfis) - min(jfis)) < 0.002 and (max(waits) - min(waits)) < 0.1:
            print("  ✓  Soft-threshold has negligible effect with these weights.")
            print("     Recommend: fix soft_threshold=0.05 and exclude from Pareto sweep.")
        else:
            print("  !  Soft-threshold has a measurable effect — include in Pareto sweep.")

    # Save CSV
    csv_keys = ["label", "strategy", "soft_threshold",
                "fairness_weight", "starvation_weight", "utility_weight", "gamma", "k",
                "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate",
                "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)", "elapsed_s"]
    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in results if not r.get("_failed"))
    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
