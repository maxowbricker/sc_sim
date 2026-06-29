#!/usr/bin/env python3
"""
Section 3 — Ablation Study
k (Candidate Pool Size) Sensitivity Sweep — Composite on Didi 20161109

Varies the k-NN candidate pool size while holding all other Composite params fixed.
Validates that k=15 is sufficient — i.e. performance saturates around k=10–15 and
larger k provides no meaningful gain (while increasing per-event cost from O(k)).

Fixed params: fairness_weight=1.0, starvation_weight=0.2, utility_weight=1.0,
              gamma=0.1, soft_threshold=0.05

Output: results/s3_ablation/k_sweep_20161109.csv

Usage:
    python scripts/experiments/s3_ablation/run_k_sweep.py
    python scripts/experiments/s3_ablation/run_k_sweep.py --output path/to/out.csv
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

DATA_ROOT   = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY  = "496528674@qq.com_20161109"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "s3_ablation")

# Fixed Composite weights — paper configuration
FIXED = dict(
    fairness_weight=1.0,
    starvation_weight=0.2,
    utility_weight=1.0,
    gamma=0.1,
    soft_threshold=0.05,
)

# k values to sweep
K_VALUES = [5, 10, 15, 20, 30]

TIMEOUT_SEC = 300

FIELDNAMES = [
    "k",
    "fairness_weight", "starvation_weight", "utility_weight", "gamma", "soft_threshold",
    "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
    "Completed", "Total", "elapsed_s",
]


# ---------------------------------------------------------------------------
# Helpers
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
        "TAR":             tar,
        "JFI (tasks)":     jfi,
        "JFI (earnings)":  jfi_earn,
        "JFI rate":        jfi_rate,
        "Avg Wait (m)":    avg_wait,
        "P95 Wait (m)":    p95_wait,
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
        "Completed":       completed,
        "Total":           total,
    }


def _sim_thread(sim, exc_holder):
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_config(workers_template, tasks_template, k: int,
               timeout_sec: float = TIMEOUT_SEC) -> Optional[Dict[str, Any]]:
    params = {**FIXED, "k": k}
    cfg = create_composite_config(assignment_strategy="composite", **params)
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
        thread = threading.Thread(target=_sim_thread, args=(sim, exc_holder), daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            return None
        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        print(f"FAILED: {exc}")
        return None

    m = extract_metrics(stats, workers)
    m["elapsed_s"] = time.time() - t0
    return m


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

COL_W  = 16
STAT_W = 13
DISPLAY_KEYS = ["TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate", "Avg Wait (m)", "P95 Wait (m)"]


def _fmt(v, key):
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def _header():
    return f"{'k':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS) + f"  {'Time (s)':>8}"


def _sep():
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 10)


def _row(k, m):
    label = f"k={k}" + ("  ← selected" if k == 15 else "")
    cells = "".join(f" {_fmt(m.get(key, 0), key):>{STAT_W}}" for key in DISPLAY_KEYS)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>8.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "k_sweep_20161109.csv")

    print("=" * 70)
    print("  Section 3 — k (Candidate Pool Size) Sensitivity Sweep")
    print(f"  Day:    {TARGET_DAY}")
    print(f"  Fixed:  fw=1.0  sw=0.2  uw=1.0  gamma=0.1  st=0.05")
    print(f"  k values: {K_VALUES}")
    print(f"  Est. time: ~{len(K_VALUES) * 90 // 60} min")
    print(f"  Output: {output_path}")
    print("=" * 70)

    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_template):,} workers | {len(tasks_template):,} tasks\n")

    results: List[Dict] = []

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for k in K_VALUES:
            label = f"k={k}" + ("  ← selected" if k == 15 else "")
            print(f"  Running {label:<18}", end="  ", flush=True)

            m = run_config(workers_template, tasks_template, k, args.timeout)
            if m is None:
                print("TIMEOUT / FAILED")
                continue

            print(
                f"TAR={m['TAR']:.4f}  "
                f"JFI={m['JFI (tasks)']:.4f}  "
                f"wait={m['Avg Wait (m)']:.2f}m  "
                f"[{m['elapsed_s']:.1f}s]"
            )
            row = {"k": k, **FIXED, **m}
            writer.writerow(row)
            f.flush()
            m["k"] = k
            results.append(m)

    print(f"\n\n{'=' * 70}")
    print("  RESULTS — k sensitivity (Composite, Didi 20161109)")
    print(f"  Fixed: fw=1.0  sw=0.2  uw=1.0  gamma=0.1  st=0.05")
    print(f"{'=' * 70}\n")
    print(_sep())
    print(_header())
    print(_sep())
    for r in results:
        print(_row(r["k"], r))
    print(_sep())

    if len(results) >= 2:
        jfis  = [r["JFI (tasks)"]  for r in results]
        waits = [r["Avg Wait (m)"] for r in results]
        tars  = [r["TAR"]          for r in results]
        times = [r["elapsed_s"]    for r in results]
        print(f"\n  Δ JFI  across k values: {max(jfis)  - min(jfis):.4f}")
        print(f"  Δ Wait across k values: {max(waits) - min(waits):.3f} m")
        print(f"  Δ TAR  across k values: {max(tars)  - min(tars):.4f}")
        print(f"  Δ Time across k values: {max(times) - min(times):.1f}s")

    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
