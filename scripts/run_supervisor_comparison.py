#!/usr/bin/env python3
"""
Supervisor Meeting — Strategy Comparison
Didi 20161109  ×  Gowalla Austin Sep 2010 (compressed, ratio 1:5)

Strategies:
    Greedy          — nearest-worker baseline (no fairness)
    k-NLF (k=15)   — fewest tasks in k-NN pool (raw count)
    k-NTF-EPH       — lowest earnings/hr in k-NN pool
    k-NTF-IR        — highest idle ratio in k-NN pool
    Composite†      — EWMA + starvation + utility (fw=1.4, sw=0.5, uw=1.0)

Output columns (matching paper table format):
    Rev. (k$)  Wait (m)  JFI ↑  TAR ↑  Backlog ↓  Time (s)

Usage:
    conda activate sc
    python3.11 scripts/run_supervisor_comparison.py
    python3.11 scripts/run_supervisor_comparison.py --didi-only
    python3.11 scripts/run_supervisor_comparison.py --gowalla-only
"""

from __future__ import annotations

import argparse
import copy
import csv as _csv
import os
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES: List[Tuple[str, str, dict]] = [
    ("Greedy",         "greedy",    {}),
    ("k-NLF (k=15)",   "knlf",      {"k": 15}),
    ("k-NTF-EPH",      "kntf_eph",  {"k": 15}),
    ("k-NTF-IR",       "kntf_ir",   {"k": 15}),
    ("Composite†",     "composite", {
        "fairness_weight":   1.4,
        "starvation_weight": 0.5,
        "utility_weight":    1.0,
        "gamma":             0.1,
        "k":                 15,
        "soft_threshold":    0.05,
    }),
]

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DIDI_ROOT     = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
DIDI_DAY      = "496528674@qq.com_20161109"

GOWALLA_ROOT  = os.path.join(PROJECT_ROOT, "data", "gowalla")
GOWALLA_KWARGS = dict(
    region                = "austin",
    date_start            = "2010-09-01",
    date_end              = "2010-09-30",
    task_mode             = "checkin",
    task_window_hours     = 0.5,
    shift_hours           = 8.0,
    dropoff_noise_km      = 2.0,
    compress_to_day       = True,
    workers_per_task_ratio= 0.20,   # 1:5
    random_state          = 42,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "supervisor")
TIMEOUT_SEC = 600   # 10 min per run


# ---------------------------------------------------------------------------
# Metric extraction
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
    revenue    = stats.get("total_platform_revenue", 0.0)

    tar        = completed / total if total else 0.0
    jfi        = stats.get("final_jains_fairness_index", 0.0)
    jfi_earn   = stats.get("final_jfi_earnings", 0.0)
    gini_tasks = stats.get("final_gini_coefficient", 0.0)
    backlog    = stats.get("backlog_peak", 0)

    avg_wait   = float(np.mean(wait_times))            if wait_times else 0.0
    p95_wait   = float(np.percentile(wait_times, 95))  if wait_times else 0.0

    counts     = sorted(w.completed_tasks for w in workers)
    jfi_rate   = sum(1 for c in counts if c > 0) / max(len(counts), 1)
    p10_tasks  = float(np.percentile(counts, 10)) if counts else 0.0
    p25_tasks  = float(np.percentile(counts, 25)) if counts else 0.0

    earnings   = [w.total_earnings for w in workers]
    idle_times = [w.total_idle_time for w in workers]
    mean_earn  = float(np.mean(earnings))   if earnings else 0.0
    std_earn   = float(np.std(earnings))    if earnings else 0.0
    mean_idle  = float(np.mean(idle_times)) if idle_times else 0.0
    std_idle   = float(np.std(idle_times))  if idle_times else 0.0
    cv_earn    = std_earn / mean_earn if mean_earn > 0 else 0.0
    cv_idle    = std_idle / mean_idle if mean_idle > 0 else 0.0
    gini_earn  = _gini(earnings)

    total_shift = sum((w.deadline - w.release_time) / 60.0 for w in workers)
    total_idle  = sum(idle_times) / 60.0
    util_pct    = 100.0 * (1.0 - total_idle / total_shift) if total_shift > 0 else 0.0

    return {
        "Completed":       completed,
        "Total":           total,
        "TAR":             tar,
        "Rev. (k$)":       revenue / 1000.0,
        # Allocation equity
        "JFI (tasks)":     jfi,
        "Gini (tasks)":    gini_tasks,
        "JFI (earnings)":  jfi_earn,
        "Gini (earn)":     gini_earn,
        "JFI rate":        jfi_rate,
        "P10 tasks":       p10_tasks,
        "P25 tasks":       p25_tasks,
        # Validation — what each signal directly optimises
        "CV (idle)":       cv_idle,
        "CV (earn)":       cv_earn,
        # Wait time
        "Wait (m)":        avg_wait,
        "P95 Wait (m)":    p95_wait,
        # Spatial & throughput
        "Backlog":         backlog,
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
        "Util (%)":        util_pct,
    }


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _sim_thread(sim, exc_holder):
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_strategy(workers_t, tasks_t, label, strat_key, params, timeout=TIMEOUT_SEC):
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
            print(f"  {label:<20}  TIMEOUT")
            return None
        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        print(f"  {label:<20}  FAILED  {exc}")
        return None

    m = extract_metrics(stats, workers)
    m["elapsed_s"] = time.time() - t0
    print(
        f"  {label:<20}  "
        f"TAR={m['TAR']:.4f}  "
        f"JFI={m['JFI (tasks)']:.4f}  "
        f"JFI-earn={m['JFI (earnings)']:.4f}  "
        f"JFI-rate={m['JFI rate']:.4f}  "
        f"wait={m['Wait (m)']:.2f}m  "
        f"backlog={m['Backlog']}  "
        f"[{m['elapsed_s']:.1f}s]"
    )
    return m


# ---------------------------------------------------------------------------
# Table display — matches paper format from image
# ---------------------------------------------------------------------------

COL_W = 18

# Primary table — paper format (matches image)
PRIMARY_COLS = ["Rev. (k$)", "Wait (m)", "JFI (tasks)", "TAR", "Backlog"]
PRIMARY_HDRS = ["Rev. (k$)", "Wait ↓",   "JFI ↑",       "TAR ↑", "Backlog ↓"]

# Block A — full allocation equity breakdown
FAIRNESS_COLS = [
    "JFI (tasks)", "Gini (tasks)",
    "JFI (earnings)", "Gini (earn)",
    "JFI rate", "P10 tasks", "P25 tasks",
]

# Block B — validation: what each signal directly optimises
VALIDATION_COLS = ["CV (idle)", "CV (earn)"]
VALIDATION_NOTE = (
    "CV(idle): lower = more uniform idle time  ← k-NTF-IR target\n"
    "  CV(earn): lower = more uniform earnings   ← k-NTF-EPH target"
)

# Block C — efficiency
EFFICIENCY_COLS = ["TAR", "Wait (m)", "P95 Wait (m)", "Avg Pickup (km)", "Util (%)", "Backlog"]

STAT_W = 12


def _fmt(v, key):
    if v is None:
        return "—"
    if isinstance(v, float):
        if key in ("JFI (tasks)", "JFI (earnings)", "JFI rate", "TAR",
                   "Gini (tasks)", "Gini (earn)"):
            return f"{v:.4f}"
        if key == "Rev. (k$)":
            return f"{v:,.1f}"
        if key in ("P10 tasks", "P25 tasks"):
            return f"{v:.1f}"
        if key in ("CV (idle)", "CV (earn)"):
            return f"{v:.4f}"
        if key in ("Util (%)",):
            return f"{v:.1f}"
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _sep(cols):
    return "─" * (COL_W + (STAT_W + 1) * len(cols) + 9)


def _header(cols, hdrs=None):
    hdrs = hdrs or cols
    return f"{'Strategy':<{COL_W}}" + "".join(f" {h:>{STAT_W}}" for h in hdrs) + f"  {'Time(s)':>7}"


def _row(label, m, cols, star=False):
    tag  = "*" if star else " "
    cells = "".join(
        f" {_fmt(m.get(c), c):>{STAT_W}}" for c in cols if c != "elapsed_s"
    )
    return f"{tag}{label:<{COL_W - 1}}{cells}  {m.get('elapsed_s', 0):>7.1f}"


def print_dataset_table(dataset_label: str, results: List[Dict]):
    W = len(_sep(FAIRNESS_COLS))  # widest block sets the border width
    print(f"\n{'═' * W}")
    print(f"  {dataset_label}")
    print(f"{'═' * W}")

    # ── Paper-format table ────────────────────────────────────────────────
    print(f"\n  ── PRIMARY (paper table format) ──")
    print(_sep(PRIMARY_COLS))
    print(_header(PRIMARY_COLS, PRIMARY_HDRS))
    print(_sep(PRIMARY_COLS))
    for r in results:
        star = "Composite" in r["_label"]
        print(_row(r["_label"], r, PRIMARY_COLS, star=star))
    print(_sep(PRIMARY_COLS))

    # ── Full allocation equity ─────────────────────────────────────────────
    print(f"\n  ── FAIRNESS (allocation equity) ──")
    print(_sep(FAIRNESS_COLS))
    print(_header(FAIRNESS_COLS))
    print(_sep(FAIRNESS_COLS))
    for r in results:
        star = "Composite" in r["_label"]
        print(_row(r["_label"], r, FAIRNESS_COLS, star=star))
    print(_sep(FAIRNESS_COLS))

    # ── Validation: what each signal directly optimises ───────────────────
    print(f"\n  ── VALIDATION (signal targets) ──")
    print(f"  {VALIDATION_NOTE}")
    print(_sep(VALIDATION_COLS))
    print(_header(VALIDATION_COLS))
    print(_sep(VALIDATION_COLS))
    for r in results:
        star = "Composite" in r["_label"]
        print(_row(r["_label"], r, VALIDATION_COLS, star=star))
    print(_sep(VALIDATION_COLS))

    # ── Efficiency ────────────────────────────────────────────────────────
    print(f"\n  ── EFFICIENCY (throughput & spatial cost) ──")
    print(_sep(EFFICIENCY_COLS))
    print(_header(EFFICIENCY_COLS))
    print(_sep(EFFICIENCY_COLS))
    for r in results:
        star = "Composite" in r["_label"]
        print(_row(r["_label"], r, EFFICIENCY_COLS, star=star))
    print(_sep(EFFICIENCY_COLS))

    print(f"\n  * = proposed method  |  k-NTF variants k=15  |  Composite fw=1.4 sw=0.5 uw=1.0")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--didi-only",    action="store_true")
    parser.add_argument("--gowalla-only", action="store_true")
    parser.add_argument("--timeout",      type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()

    run_didi    = not args.gowalla_only
    run_gowalla = not args.didi_only

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")

    print("=" * 72)
    print("  Supervisor Meeting — Strategy Comparison")
    print(f"  Composite weights: fw=1.4  sw=0.5  uw=1.0  k=15  γ=0.1")
    print(f"  Strategies: {', '.join(s[0] for s in STRATEGIES)}")
    datasets = (["Didi 20161109"] if run_didi else []) + (["Gowalla Austin (compressed, 1:5)"] if run_gowalla else [])
    print(f"  Datasets:   {', '.join(datasets)}")
    print(f"  Timeout:    {args.timeout:.0f}s / run")
    print(f"  Output:     {RESULTS_DIR}/")
    print("=" * 72)

    all_csv_rows = []
    all_tables   = []

    # ── Gowalla first (faster, good for a quick sanity check) ─────────────
    if run_gowalla:
        print(f"\n{'━' * 72}")
        print("  GOWALLA  Austin Sep 2010  (compressed, ratio 1:5)")
        print(f"{'━' * 72}")
        workers_t, tasks_t = load_workers_tasks(
            "gowalla", root_path=GOWALLA_ROOT, **GOWALLA_KWARGS
        )
        print(f"  {len(workers_t):,} workers | {len(tasks_t):,} tasks\n")

        gowalla_results = []
        for label, key, params in STRATEGIES:
            m = run_strategy(workers_t, tasks_t, label, key, params, args.timeout)
            if m is None:
                continue
            m["_label"]   = label
            m["_dataset"] = "Gowalla"
            gowalla_results.append(m)
            all_csv_rows.append(m)

        all_tables.append(("GOWALLA  Austin Sep 2010  (compressed, 1:5)", gowalla_results))

    # ── Didi ──────────────────────────────────────────────────────────────
    if run_didi:
        day_path = os.path.join(DIDI_ROOT, DIDI_DAY)
        print(f"\n{'━' * 72}")
        print(f"  DIDI  {DIDI_DAY}")
        print(f"{'━' * 72}")
        workers_t, tasks_t = load_workers_tasks("didi", root_path=day_path)
        print(f"  {len(workers_t):,} workers | {len(tasks_t):,} tasks\n")

        didi_results = []
        for label, key, params in STRATEGIES:
            m = run_strategy(workers_t, tasks_t, label, key, params, args.timeout)
            if m is None:
                continue
            m["_label"]   = label
            m["_dataset"] = "Didi"
            didi_results.append(m)
            all_csv_rows.append(m)

        all_tables.append(("DIDI  20161109", didi_results))

    # ── Summary ───────────────────────────────────────────────────────────
    for dataset_label, results in all_tables:
        print_dataset_table(dataset_label, results)

    # ── CSV ───────────────────────────────────────────────────────────────
    if all_csv_rows:
        csv_path = os.path.join(RESULTS_DIR, f"supervisor_comparison_{ts}.csv")
        fieldnames = [
            "_dataset", "_label",
            "TAR", "Rev. (k$)",
            "JFI (tasks)", "Gini (tasks)",
            "JFI (earnings)", "Gini (earn)",
            "JFI rate", "P10 tasks", "P25 tasks",
            "CV (idle)", "CV (earn)",
            "Wait (m)", "P95 Wait (m)",
            "Avg Pickup (km)", "Util (%)", "Backlog",
            "elapsed_s",
        ]
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_csv_rows)
        print(f"\n  CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
