#!/usr/bin/env python3
"""
Section 5.4.2 — Ablation & Sensitivity Analysis: Fairness Signal Comparison
k=15, Didi 20161109

Compares five O(k) or O(W) fairness signals applied on top of the same
k=15 spatial candidate pool, plus Greedy as the no-fairness anchor:

    Signal          Strategy    Sort key in k-NN pool
    ──────────────  ──────────  ─────────────────────────────────────────────
    None            Greedy      Nearest worker (distance only)
    EWMA idle time  EWMA-Only   Highest EWMA idle-time signal (raw seconds)
    Task count      k-NLF       Fewest completed_tasks (raw count)
    Earnings/hr     k-NTF-EPH   Lowest total_earnings / shift_elapsed_hours
    Idle ratio      k-NTF-IR    Highest total_idle_time / shift_elapsed
    Unconstrained   LAF         Fewest completed_tasks across ALL workers (O(W))

This directly answers the paper's central question:
  "Which O(k) fairness signal best balances JFI improvement against wait-time cost?"

Output: results/s54_ablation/signal_comparison_20161109.csv

Usage:
    python scripts/experiments/s54_ablation/run_signal_comparison.py
    python scripts/experiments/s54_ablation/run_signal_comparison.py --output path/out.csv
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

# (display_label, strategy_key, params, complexity_note)
STRATEGIES: List[Tuple[str, str, dict, str]] = [
    # --- Lower bound: no fairness ---
    ("Greedy",           "greedy",    {},               "O(W) scan, nearest wins"),
    # --- O(k) signals (same k=15 spatial pool, different sort key) ---
    ("k-NLF (k=15)",     "knlf",      {"k": 15},        "O(k): fewest tasks (raw count)"),
    ("k-NTF-EPH (k=15)", "kntf_eph",  {"k": 15},        "O(k): lowest earnings/hr"),
    ("k-NTF-EPH (k=15)", "kntf_eph",  {"k": 5},        "O(k): lowest earnings/hr")
    ,("k-NTF-IR  (k=15)", "kntf_ir",   {"k": 15},        "O(k): highest idle ratio"),
    ("k-NTF-IR  (k=15)", "kntf_ir",   {"k": 5},        "O(k): highest idle ratio")
]

TIMEOUT_SEC = 600

FIELDNAMES = [
    "strategy", "complexity",
    "TAR", "Revenue ($)", "JFI (tasks)", "Gini (tasks)", "JFI (earnings)", "Gini (earn)",
    "JFI rate", "P10 tasks", "P25 tasks",
    "CV (idle)", "CV (earn)",
    "Avg Wait (m)", "P50 Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
    "Completed", "Total", "elapsed_s",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(values):
    """Population Gini coefficient for a list of non-negative floats."""
    arr = np.array(values, dtype=float)
    if arr.sum() == 0 or len(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def extract_metrics(stats: Dict[str, Any], workers) -> Dict[str, Any]:
    completed  = stats.get("completed_tasks", 0)
    total      = stats.get("total_tasks", 1)
    wait_times = stats.get("wait_times", [])

    tar      = completed / total if total else 0.0
    revenue  = stats.get("total_platform_revenue", 0.0)
    jfi      = stats.get("final_jains_fairness_index", 0.0)
    gini     = stats.get("final_gini_coefficient", 0.0)
    jfi_earn = stats.get("final_jfi_earnings", 0.0)

    counts   = sorted(w.completed_tasks for w in workers)
    jfi_rate = sum(1 for c in counts if c > 0) / max(len(counts), 1)
    p10      = float(np.percentile(counts, 10)) if counts else 0.0
    p25      = float(np.percentile(counts, 25)) if counts else 0.0

    avg_wait = float(np.mean(wait_times))            if wait_times else 0.0
    p50_wait = float(np.percentile(wait_times, 50))  if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95))  if wait_times else 0.0

    # --- Validation metrics: each strategy directly optimises one of these ---

    # k-NTF-IR claim: reduces idle-time inequality
    idle_times = [w.total_idle_time for w in workers]
    mean_idle  = float(np.mean(idle_times)) if idle_times else 0.0
    std_idle   = float(np.std(idle_times))  if idle_times else 0.0
    cv_idle    = std_idle / mean_idle if mean_idle > 0 else 0.0

    # k-NTF-EPH claim: reduces earnings inequality
    earnings   = [w.total_earnings for w in workers]
    mean_earn  = float(np.mean(earnings)) if earnings else 0.0
    std_earn   = float(np.std(earnings))  if earnings else 0.0
    cv_earn    = std_earn / mean_earn if mean_earn > 0 else 0.0
    gini_earn  = _gini(earnings)

    return {
        "TAR":             tar,
        "Revenue ($)":     revenue,
        "JFI (tasks)":     jfi,
        "Gini (tasks)":    gini,
        "JFI (earnings)":  jfi_earn,
        "Gini (earn)":     gini_earn,
        "JFI rate":        jfi_rate,
        "P10 tasks":       p10,
        "P25 tasks":       p25,
        "CV (idle)":       cv_idle,
        "CV (earn)":       cv_earn,
        "Avg Wait (m)":    avg_wait,
        "P50 Wait (m)":    p50_wait,
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


def run_config(workers_t, tasks_t, strat_key, params, timeout):
    cfg = create_composite_config(assignment_strategy=strat_key, **params)
    t0  = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator
        sim = EventSimulator(copy.deepcopy(workers_t), copy.deepcopy(tasks_t), cfg)
        sim.reset()
        thread = threading.Thread(target=_sim_thread, args=(sim, exc_holder), daemon=True)
        thread.start()
        thread.join(timeout=timeout)

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
# Display
# ---------------------------------------------------------------------------

COL_W  = 22
STAT_W = 12

# Block 1: Standard fairness axes (allocation equity)
FAIRNESS_COLS = [
    "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Gini (tasks)", "P10 tasks", "P25 tasks",
]

# Block 2: Validation metrics — each signal directly optimises one of these
# CV (idle)  ← k-NTF-IR should win here (lower = more uniform idle time)
# CV (earn)  ← k-NTF-EPH should win here (lower = more uniform earnings)
# Gini(earn) ← k-NTF-EPH second validation
VALIDATION_COLS = [
    "CV (idle)", "CV (earn)", "Gini (earn)",
]

# Block 3: Efficiency / spatial cost + platform revenue
EFFICIENCY_COLS = [
    "TAR", "Revenue ($)", "Avg Wait (m)", "P50 Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
]


def _fmt(v, key):
    if isinstance(v, float):
        if key in ("JFI (tasks)", "JFI (earnings)", "JFI rate",
                   "Gini (tasks)", "Gini (earn)", "TAR"):
            return f"{v:.4f}"
        if key in ("P10 tasks", "P25 tasks"):
            return f"{v:.1f}"
        if key in ("CV (idle)", "CV (earn)"):
            return f"{v:.4f}"
        if key == "Revenue ($)":
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)


def _header(cols):
    return (f"{'Strategy':<{COL_W}}"
            + "".join(f" {k:>{STAT_W}}" for k in cols)
            + f"  {'Time(s)':>7}")


def _sep(cols):
    return "─" * (COL_W + (STAT_W + 1) * len(cols) + 9)


def _row(label, m, cols):
    cells = "".join(f" {_fmt(m.get(k, 0), k):>{STAT_W}}" for k in cols)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>7.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",  type=str,   default=None)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "signal_comparison_20161109.csv")

    print("=" * 75)
    print("  Section 5.4.2 — Fairness Signal Comparison (k=15, Didi 20161109)")
    print()
    print("  Signal          Strategy")
    for label, key, _, note in STRATEGIES:
        print(f"    {label:<20}  {note}")
    print(f"\n  Total runs: {len(STRATEGIES)}")
    print(f"  Est. time:  ~{len(STRATEGIES) * 6} min")
    print(f"  Output:     {output_path}")
    print("=" * 75)

    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    workers_t, tasks_t = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_t):,} workers | {len(tasks_t):,} tasks\n")

    all_results: List[Dict] = []

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for idx, (label, strat_key, params, note) in enumerate(STRATEGIES, 1):
            print(f"  [{idx}/{len(STRATEGIES)}]  {label:<22}", end="  ", flush=True)

            m = run_config(workers_t, tasks_t, strat_key, params, args.timeout)
            if m is None:
                print("TIMEOUT / FAILED")
                continue

            print(
                f"TAR={m['TAR']:.4f}  "
                f"JFI(t)={m['JFI (tasks)']:.4f}  "
                f"JFI(e)={m['JFI (earnings)']:.4f}  "
                f"JFI-rate={m['JFI rate']:.4f}  "
                f"CV(idle)={m['CV (idle)']:.4f}  "
                f"CV(earn)={m['CV (earn)']:.4f}  "
                f"wait={m['Avg Wait (m)']:.2f}m  "
                f"[{m['elapsed_s']:.1f}s]"
            )

            writer.writerow({"strategy": label, "complexity": note, **m})
            f.flush()
            m["_label"] = label
            all_results.append(m)

    # Summary
    W = COL_W + (STAT_W + 1) * len(FAIRNESS_COLS) + 9
    print(f"\n\n{'=' * W}")
    print("  RESULTS — Fairness Signal Comparison")
    print(f"  k-NN variants at k=15 | Didi 20161109")
    print(f"{'=' * W}")

    print(f"\n  ── FAIRNESS (allocation equity) ──")
    print(_sep(FAIRNESS_COLS))
    print(_header(FAIRNESS_COLS))
    print(_sep(FAIRNESS_COLS))
    for r in all_results:
        print(_row(r["_label"], r, FAIRNESS_COLS))
    print(_sep(FAIRNESS_COLS))

    print(f"\n  ── VALIDATION (what each signal directly optimises) ──")
    print(f"     CV(idle): lower = more uniform idle distribution  ← k-NTF-IR target")
    print(f"     CV(earn): lower = more uniform earnings           ← k-NTF-EPH target")
    print(_sep(VALIDATION_COLS))
    print(_header(VALIDATION_COLS))
    print(_sep(VALIDATION_COLS))
    for r in all_results:
        print(_row(r["_label"], r, VALIDATION_COLS))
    print(_sep(VALIDATION_COLS))

    print(f"\n  ── EFFICIENCY (spatial & throughput cost) ──")
    print(_sep(EFFICIENCY_COLS))
    print(_header(EFFICIENCY_COLS))
    print(_sep(EFFICIENCY_COLS))
    for r in all_results:
        print(_row(r["_label"], r, EFFICIENCY_COLS))
    print(_sep(EFFICIENCY_COLS))

    # Delta table vs k-NLF (same structural approach, different signal)
    knlf = next((r for r in all_results if "k-NLF" in r["_label"]), None)
    if knlf:
        print(f"\n  Δ vs k-NLF (JFI-tasks={knlf['JFI (tasks)']:.4f}  JFI-earn={knlf['JFI (earnings)']:.4f}  "
              f"JFI-rate={knlf['JFI rate']:.4f}  wait={knlf['Avg Wait (m)']:.3f}m):")
        for r in all_results:
            if "k-NLF" in r["_label"]:
                continue
            dj  = r["JFI (tasks)"]    - knlf["JFI (tasks)"]
            djer = r["JFI (earnings)"] - knlf["JFI (earnings)"]
            djr = r["JFI rate"]        - knlf["JFI rate"]
            dw  = r["Avg Wait (m)"]   - knlf["Avg Wait (m)"]
            dp10 = r["P10 tasks"]     - knlf["P10 tasks"]
            _s  = lambda x: ("+" if x >= 0 else "")
            print(
                f"    {r['_label']:<24}"
                f"  ΔJ-tasks={_s(dj)}{dj:.4f}"
                f"  ΔJ-earn={_s(djer)}{djer:.4f}"
                f"  ΔJ-rate={_s(djr)}{djr:.4f}"
                f"  ΔWait={_s(dw)}{dw:.3f}m"
                f"  ΔP10={_s(dp10)}{dp10:.1f}"
            )

    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
