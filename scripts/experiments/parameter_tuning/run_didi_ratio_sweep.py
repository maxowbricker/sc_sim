#!/usr/bin/env python3
"""
Section 4 — Robustness to Spatiotemporal Density
Worker:Task Ratio Sweep — Didi 20161109

Varies the worker:task ratio by subsampling loaded workers, running 4 key strategies
at each ratio. Complements the Gowalla ratio sweep (which varies ratio via the loader).

Ratios tested: 1:4 (0.25), 1:5 (0.20), 1:7 (0.143)
Strategies: Greedy, LAF, ONRTA-OP, Composite (static)

Workers are randomly subsampled from the full 36,799 to match the target ratio
relative to task count. Tasks are kept at full volume (224,219).

Output: results/s4_robustness/didi_ratio_sweep.csv

Usage:
    python scripts/experiments/s4_robustness/run_didi_ratio_sweep.py
    python scripts/experiments/s4_robustness/run_didi_ratio_sweep.py --output path/to/out.csv
    python scripts/experiments/s4_robustness/run_didi_ratio_sweep.py --seed 99  # different subsample
"""

from __future__ import annotations

import argparse
import copy
import csv as _csv
import os
import random
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "s4_robustness")

# (label, workers_per_task_ratio)
RATIO_CONFIGS: List[Tuple[str, float]] = [
    ("1:4",  0.25),
    ("1:5",  0.20),
    ("1:7",  1 / 7),
]

# Strategies — order by ascending wall time
STRATEGIES: List[Tuple[str, str, dict]] = [
    ("Greedy",          "greedy", {}),
    ("LAF",             "laf",    {}),
    ("ONRTA-OP",        "onrta_op", {}),
    ("Composite",       "composite", {
        "fairness_weight": 1.0,
        "starvation_weight": 0.2,
        "utility_weight": 1.0,
        "gamma": 0.1,
        "k": 15,
        "soft_threshold": 0.05,
    }),
]

TIMEOUT_SEC = 600  # 10 min — larger pool may be slower

FIELDNAMES = [
    "ratio_label", "ratio_value", "n_workers", "n_tasks",
    "strategy",
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


def run_config(
    workers_subset: list,
    tasks_template: list,
    strategy_key: str,
    strategy_params: dict,
    timeout_sec: float = TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    cfg = create_composite_config(assignment_strategy=strategy_key, **strategy_params)
    t0  = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator
        sim = EventSimulator(
            copy.deepcopy(workers_subset),
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

COL_W  = 28
STAT_W = 13
DISPLAY_KEYS = ["TAR", "JFI (tasks)", "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)"]


def _fmt(v, key):
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def _header():
    return f"{'Ratio / Strategy':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS) + f"  {'Time (s)':>8}"


def _sep():
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 10)


def _row(label, m):
    cells = "".join(f" {_fmt(m.get(k, 0), k):>{STAT_W}}" for k in DISPLAY_KEYS)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>8.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for worker subsampling (default: 42)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "didi_ratio_sweep.csv")

    n_ratios     = len(RATIO_CONFIGS)
    n_strategies = len(STRATEGIES)
    n_total      = n_ratios * n_strategies

    print("=" * 75)
    print("  Section 4 — Didi Worker:Task Ratio Sweep")
    print(f"  Day:       {TARGET_DAY}")
    print(f"  Ratios:    {', '.join(r[0] for r in RATIO_CONFIGS)}")
    print(f"  Strategies: {', '.join(s[0] for s in STRATEGIES)}")
    print(f"  Total runs: {n_total}")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {output_path}")
    print("=" * 75)

    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    all_workers, all_tasks = load_workers_tasks("didi", root_path=day_path)
    n_tasks = len(all_tasks)
    print(f"  {len(all_workers):,} workers | {n_tasks:,} tasks  (full load)\n")

    rng = random.Random(args.seed)
    all_results: List[Dict] = []

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        run_idx = 0
        for ratio_label, ratio in RATIO_CONFIGS:
            n_workers_target = min(int(n_tasks * ratio), len(all_workers))
            workers_subset = rng.sample(all_workers, n_workers_target)

            print(f"\n  {'─' * 71}")
            print(f"  Ratio {ratio_label}  (workers = {n_workers_target:,}  tasks = {n_tasks:,}  ratio ≈ {ratio:.4f})")
            print(f"  {'─' * 71}")

            for strat_name, strat_key, strat_params in STRATEGIES:
                run_idx += 1
                label = f"{ratio_label} | {strat_name}"
                print(f"  [{run_idx:>2}/{n_total}]  {label:<26}", end="  ", flush=True)

                m = run_config(
                    workers_subset, all_tasks,
                    strat_key, strat_params,
                    args.timeout,
                )

                if m is None:
                    print("TIMEOUT / FAILED")
                    continue

                print(
                    f"TAR={m['TAR']:.4f}  "
                    f"JFI={m['JFI (tasks)']:.4f}  "
                    f"wait={m['Avg Wait (m)']:.2f}m  "
                    f"[{m['elapsed_s']:.1f}s]"
                )

                row = {
                    "ratio_label": ratio_label,
                    "ratio_value": ratio,
                    "n_workers":   n_workers_target,
                    "n_tasks":     n_tasks,
                    "strategy":    strat_name,
                    **m,
                }
                writer.writerow(row)
                f.flush()
                m.update({"ratio_label": ratio_label, "strategy": strat_name})
                all_results.append(m)

    # Summary
    print(f"\n\n{'=' * 75}")
    print("  FINAL RESULTS — Didi Ratio Sweep")
    print(f"{'=' * 75}\n")
    print(_sep())
    print(_header())
    print(_sep())
    prev_ratio = None
    for r in all_results:
        if r["ratio_label"] != prev_ratio:
            if prev_ratio is not None:
                print()
            prev_ratio = r["ratio_label"]
        label = f"{r['ratio_label']} | {r['strategy']}"
        print(_row(label, r))
    print(_sep())

    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
