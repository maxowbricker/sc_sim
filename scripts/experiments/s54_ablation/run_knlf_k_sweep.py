#!/usr/bin/env python3
"""
Section 5.4.1 — Ablation & Sensitivity Analysis
k-NLF Candidate Pool Size (k) Sweep — Didi 20161109

Sweeps k ∈ {3, 5, 10, 15} for the k-Nearest Least-First strategy to show the
fairness–throughput tradeoff as the spatial constraint is relaxed:

  k=3   → very tight spatial constraint (almost as close as Greedy)
  k=5   → small relaxation
  k=10  → moderate relaxation
  k=15  → paper default (same candidate pool as Composite)

Also runs Greedy (k=∞ but nearest-only) and LAF (k=W, all workers) as anchors
so the entire tradeoff curve is visible in a single plot.

Output: results/s54_ablation/knlf_k_sweep_20161109.csv

Usage:
    python scripts/experiments/s54_ablation/run_knlf_k_sweep.py
    python scripts/experiments/s54_ablation/run_knlf_k_sweep.py --output path/to/out.csv
"""

from __future__ import annotations

import argparse
import copy
import csv as _csv
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

DATA_ROOT   = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY  = "496528674@qq.com_20161109"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "s54_ablation")

# Anchor strategies for context on the tradeoff curve
ANCHORS: List[Tuple[str, str, dict]] = [
    ("Greedy (k=∞, dist)",  "greedy", {}),
    ("LAF (k=W, count)",    "laf",    {}),
]

# k-NLF sweep configs: (display_label, k)
K_VALUES = [3, 5, 10, 15, 25, 50, 100]

TIMEOUT_SEC = 300

FIELDNAMES = [
    "label", "strategy", "k",
    "TAR", "Revenue ($)", "JFI (tasks)", "Gini (tasks)", "JFI (earnings)", "JFI rate",
    "P10 tasks", "P25 tasks",
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
    revenue  = stats.get("total_platform_revenue", 0.0)
    jfi      = stats.get("final_jains_fairness_index", 0.0)
    gini     = stats.get("final_gini_coefficient", 0.0)
    jfi_earn = stats.get("final_jfi_earnings", 0.0)

    worker_task_counts = sorted(w.completed_tasks for w in workers)
    jfi_rate  = sum(1 for c in worker_task_counts if c > 0) / max(len(worker_task_counts), 1)
    p10_tasks = float(np.percentile(worker_task_counts, 10)) if worker_task_counts else 0.0
    p25_tasks = float(np.percentile(worker_task_counts, 25)) if worker_task_counts else 0.0

    avg_wait = float(np.mean(wait_times))            if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95))  if wait_times else 0.0

    return {
        "TAR":             tar,
        "Revenue ($)":     revenue,
        "JFI (tasks)":     jfi,
        "Gini (tasks)":    gini,
        "JFI (earnings)":  jfi_earn,
        "JFI rate":        jfi_rate,
        "P10 tasks":       p10_tasks,
        "P25 tasks":       p25_tasks,
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


def run_config(
    workers_template, tasks_template,
    strategy_key: str, params: dict,
    timeout_sec: float = TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
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

COL_W  = 22
STAT_W = 12
DISPLAY_KEYS = ["TAR", "Revenue ($)", "JFI (tasks)", "Gini (tasks)", "JFI rate", "P10 tasks", "Avg Wait (m)", "P95 Wait (m)"]


def _fmt(v, key):
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "Gini (tasks)", "JFI rate"):
            return f"{v:.4f}"
        if key in ("P10 tasks", "P25 tasks"):
            return f"{v:.1f}"
        if key == "Revenue ($)":
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)


def _header():
    return f"{'Strategy':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS) + f"  {'Time(s)':>7}"


def _sep():
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 9)


def _row(label, m):
    cells = "".join(f" {_fmt(m.get(k, 0), k):>{STAT_W}}" for k in DISPLAY_KEYS)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>7.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "knlf_k_sweep_20161109.csv")

    n_total = len(ANCHORS) + len(K_VALUES)
    print("=" * 75)
    print("  Section 5.4.1 — k-NLF Candidate Pool Size (k) Sweep")
    print(f"  Day:    {TARGET_DAY}")
    print(f"  k values: {K_VALUES}  +  Greedy and LAF anchors")
    print(f"  Total runs: {n_total}")
    print(f"  Est. time: ~{n_total * 5} min")
    print(f"  Output: {output_path}")
    print("=" * 75)

    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_template):,} workers | {len(tasks_template):,} tasks\n")

    all_results: List[Dict] = []

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        # --- Anchor strategies ---
        for label, strat_key, params in ANCHORS:
            k_val = "W" if strat_key == "laf" else "∞"
            print(f"  Running {label:<28}", end="  ", flush=True)
            m = run_config(workers_template, tasks_template, strat_key, params, args.timeout)
            if m is None:
                print("TIMEOUT / FAILED")
                continue
            print(
                f"TAR={m['TAR']:.4f}  JFI={m['JFI (tasks)']:.4f}  "
                f"wait={m['Avg Wait (m)']:.2f}m  [{m['elapsed_s']:.1f}s]"
            )
            row = {"label": label, "strategy": strat_key, "k": k_val, **m}
            writer.writerow(row)
            f.flush()
            m["label"] = label
            all_results.append(m)

        print()

        # --- k-NLF sweep ---
        for k in K_VALUES:
            label = f"k-NLF (k={k})" + ("  ← default" if k == 15 else "")
            print(f"  Running {label:<28}", end="  ", flush=True)
            m = run_config(
                workers_template, tasks_template,
                "knlf", {"k": k},
                args.timeout,
            )
            if m is None:
                print("TIMEOUT / FAILED")
                continue
            print(
                f"TAR={m['TAR']:.4f}  JFI={m['JFI (tasks)']:.4f}  "
                f"wait={m['Avg Wait (m)']:.2f}m  [{m['elapsed_s']:.1f}s]"
            )
            row = {"label": f"k-NLF k={k}", "strategy": "knlf", "k": k, **m}
            writer.writerow(row)
            f.flush()
            m["label"] = label
            all_results.append(m)

    # Summary table
    print(f"\n\n{'=' * 75}")
    print("  RESULTS — k-NLF k sweep vs anchors (Didi 20161109)")
    print(f"{'=' * 75}\n")
    print(_sep())
    print(_header())
    print(_sep())
    # Print anchors first, then sweep
    for r in all_results:
        print(_row(r["label"], r))
    print(_sep())

    # Highlight the fairness gain at each k vs Greedy
    greedy_row = next((r for r in all_results if "Greedy" in r["label"]), None)
    if greedy_row:
        print(f"\n  JFI gain vs Greedy (JFI={greedy_row['JFI (tasks)']:.4f}):")
        for r in all_results:
            if "Greedy" in r["label"]:
                continue
            delta = r["JFI (tasks)"] - greedy_row["JFI (tasks)"]
            wait_delta = r["Avg Wait (m)"] - greedy_row["Avg Wait (m)"]
            sign = "+" if delta >= 0 else ""
            print(f"    {r['label']:<28}  ΔJFI={sign}{delta:.4f}  ΔWait={wait_delta:+.3f}m")

    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
