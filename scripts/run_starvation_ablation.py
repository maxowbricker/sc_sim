#!/usr/bin/env python3
"""
Starvation Ablation Test
========================
Tests the hypothesis: "The starvation_weight component of Composite does not
improve fairness — its only job is backlog control."

Comparing on both Didi 20161109 and Gowalla Austin Sep 2010 (compressed, 1:5):

    Strategy                  sw    Notes
    ─────────────────────────────────────────────────────────────────────
    Greedy                    —     Anchor (no fairness, greedy FREE_WORKER)
    k-NLF (k=15)              —     Best O(k) fairness signal
    Composite (fw=1.4, sw=0.5)  0.5   Current proposed config (with starvation)
    Composite (fw=1.4, sw=0.0)  0.0   Starvation removed; FREE_WORKER → greedy

Key questions:
    1. Does JFI change materially when sw is dropped?
    2. Does backlog increase when sw=0 (confirming starvation's role)?
    3. Does wait time change?

If JFI is unchanged but backlog increases → starvation term is only for
backlog control, not fairness. The paper can then cleanly separate the two
concerns: fairness (NEW_TASK signal) vs throughput (FREE_WORKER strategy).

Usage:
    conda activate sc
    python3.11 scripts/run_starvation_ablation.py
    python3.11 scripts/run_starvation_ablation.py --didi-only
    python3.11 scripts/run_starvation_ablation.py --gowalla-only
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
# Strategies — ordered from cheapest to most expensive
# ---------------------------------------------------------------------------

COMPOSITE_BASE = dict(
    fairness_weight=1.4,
    utility_weight=1.0,
    gamma=0.1,
    k=15,
    soft_threshold=0.05,
)

STRATEGIES: List[Tuple[str, str, dict]] = [
    ("Greedy",                    "greedy",    {}),
    ("k-NLF (k=15)",              "knlf",      {"k": 15}),
    ("Composite sw=0.0 (no starv)", "composite", {**COMPOSITE_BASE, "starvation_weight": 0.0}),
    ("Composite sw=0.5 (current)",  "composite", {**COMPOSITE_BASE, "starvation_weight": 0.5}),
]

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DIDI_ROOT  = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
DIDI_DAY   = "496528674@qq.com_20161109"

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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "starvation_ablation")
TIMEOUT_SEC = 600


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _jfi(values):
    """Jain's Fairness Index: (sum x_i)^2 / (n * sum x_i^2).
    Returns 1.0 for perfect equality, 1/n for maximally concentrated."""
    arr = np.array(values, dtype=float)
    if arr.sum() == 0 or len(arr) == 0:
        return 0.0
    return float(arr.sum() ** 2 / (len(arr) * (arr ** 2).sum()))


def _gini(values):
    """Population Gini coefficient. 0 = perfect equality, 1 = maximal inequality."""
    arr = np.sort(np.array(values, dtype=float))
    if arr.sum() == 0 or len(arr) == 0:
        return 0.0
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def extract_metrics(stats: Dict[str, Any], workers) -> Dict[str, Any]:
    completed  = stats.get("completed_tasks", 0)
    total      = stats.get("total_tasks", 1)
    wait_times = stats.get("wait_times", [])
    revenue    = stats.get("total_platform_revenue", 0.0)

    tar      = completed / total if total else 0.0
    jfi      = stats.get("final_jains_fairness_index", 0.0)
    jfi_earn = stats.get("final_jfi_earnings", 0.0)
    backlog  = stats.get("backlog_peak", 0)

    counts   = sorted(w.completed_tasks for w in workers)
    jfi_rate = sum(1 for c in counts if c > 0) / max(len(counts), 1)
    p10      = float(np.percentile(counts, 10)) if counts else 0.0

    avg_wait = float(np.mean(wait_times))            if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95))  if wait_times else 0.0

    idle = [w.total_idle_time for w in workers]
    earn = [w.total_earnings  for w in workers]

    # --- Idle-time equity (Composite's EWMA directly optimises this) ---
    jfi_idle  = _jfi(idle)      # higher = more uniform idle distribution
    gini_idle = _gini(idle)     # lower  = more uniform idle distribution

    cv_idle   = float(np.std(idle)) / float(np.mean(idle)) if np.mean(idle) > 0 else 0.0
    cv_earn   = float(np.std(earn)) / float(np.mean(earn)) if np.mean(earn) > 0 else 0.0

    return {
        "Completed":       completed,
        "Total":           total,
        "TAR":             tar,
        "Rev. (k$)":       revenue / 1000.0,
        # Task-count equity (what JFI paper tables usually report)
        "JFI (tasks)":     jfi,
        "JFI (earnings)":  jfi_earn,
        "JFI rate":        jfi_rate,
        "P10 tasks":       p10,
        # Idle-time equity (what Composite's EWMA signal targets)
        "JFI (idle)":      jfi_idle,
        "Gini (idle)":     gini_idle,
        "CV (idle)":       cv_idle,
        # Earnings equity
        "CV (earn)":       cv_earn,
        # Efficiency
        "Wait (m)":        avg_wait,
        "P95 Wait (m)":    p95_wait,
        "Backlog":         backlog,
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
            print(f"  {label:<36}  TIMEOUT")
            return None
        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        print(f"  {label:<36}  FAILED  {exc}")
        return None

    m = extract_metrics(stats, workers)
    m["elapsed_s"] = time.time() - t0
    m["_deferral"] = stats.get("deferral_stats", {})
    print(
        f"  {label:<36}  "
        f"JFI(t)={m['JFI (tasks)']:.4f}  "
        f"JFI(idle)={m['JFI (idle)']:.4f}  "
        f"Gini(idle)={m['Gini (idle)']:.4f}  "
        f"wait={m['Wait (m)']:.2f}m  "
        f"backlog={m['Backlog']:>4}  "
        f"[{m['elapsed_s']:.1f}s]"
    )
    return m


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

COL_W  = 30
STAT_W = 11

# Block A — task-count equity (standard paper metric)
TASK_EQUITY = ["JFI (tasks)", "JFI (earnings)", "JFI rate", "P10 tasks"]

# Block B — idle-time equity (Composite's EWMA directly targets this)
# Higher JFI(idle) = more uniform idle distribution = better temporal fairness
# Lower  Gini(idle) / CV(idle) = same interpretation
IDLE_EQUITY = ["JFI (idle)", "Gini (idle)", "CV (idle)"]

# Block C — efficiency + backlog (Composite's starvation term targets this)
EFFICIENCY  = ["Wait (m)", "P95 Wait (m)", "Backlog", "TAR", "Rev. (k$)"]


def _fmt(v, key):
    if v is None:
        return "—"
    if isinstance(v, float):
        if key in ("JFI (tasks)", "JFI (earnings)", "JFI rate", "TAR",
                   "JFI (idle)", "Gini (idle)"):
            return f"{v:.4f}"
        if key in ("CV (idle)", "CV (earn)"):
            return f"{v:.4f}"
        if key in ("P10 tasks",):
            return f"{v:.1f}"
        if key == "Rev. (k$)":
            return f"{v:,.1f}"
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _sep(cols):
    return "─" * (COL_W + (STAT_W + 1) * len(cols) + 9)


def _hdr(cols):
    return f"{'Strategy':<{COL_W}}" + "".join(f" {c:>{STAT_W}}" for c in cols) + f"  {'Time(s)':>7}"


def _row(label, m, cols):
    cells = "".join(f" {_fmt(m.get(c), c):>{STAT_W}}" for c in cols)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>7.1f}"


def print_results(dataset_label: str, results: List[Dict]):
    W = max(len(_sep(TASK_EQUITY)), len(_sep(IDLE_EQUITY)), len(_sep(EFFICIENCY)))
    print(f"\n{'═' * W}")
    print(f"  {dataset_label}")
    print(f"{'═' * W}")

    # ── Task-count equity ─────────────────────────────────────────────────
    print(f"\n  ── TASK-COUNT EQUITY (standard JFI — what all papers report) ──")
    print(_sep(TASK_EQUITY))
    print(_hdr(TASK_EQUITY))
    print(_sep(TASK_EQUITY))
    for r in results:
        print(_row(r["_label"], r, TASK_EQUITY))
    print(_sep(TASK_EQUITY))

    # ── Idle-time equity ──────────────────────────────────────────────────
    print(f"\n  ── IDLE-TIME EQUITY (Composite's EWMA directly targets this) ──")
    print(f"     JFI(idle) ↑ higher = idle time more equally distributed across workers")
    print(f"     Gini(idle) ↓ lower = same interpretation from the other direction")
    print(_sep(IDLE_EQUITY))
    print(_hdr(IDLE_EQUITY))
    print(_sep(IDLE_EQUITY))
    for r in results:
        print(_row(r["_label"], r, IDLE_EQUITY))
    print(_sep(IDLE_EQUITY))

    # ── Efficiency + backlog ──────────────────────────────────────────────
    print(f"\n  ── EFFICIENCY (Composite's starvation term targets Backlog) ──")
    print(_sep(EFFICIENCY))
    print(_hdr(EFFICIENCY))
    print(_sep(EFFICIENCY))
    for r in results:
        print(_row(r["_label"], r, EFFICIENCY))
    print(_sep(EFFICIENCY))

    # ── Deferral lifecycle ────────────────────────────────────────────────
    print(f"\n  ── DEFERRAL LIFECYCLE (deferred-then-rescued vs deferred-then-expired) ──")
    print(f"  {'Strategy':<34} {'Deferred':>9} {'Rescued':>9} {'Expired':>9} {'Pending':>9} {'Rescue%':>8}  Time(s)")
    print("  " + "─" * 78)
    for r in results:
        d = r.get("_deferral", {})
        deferred = d.get("unique_tasks_deferred", 0)
        rescued  = d.get("deferred_then_rescued",  0)
        expired  = d.get("deferred_then_expired",  0)
        pending  = d.get("deferred_still_pending", 0)
        rate     = d.get("rescue_rate", 0.0)
        print(
            f"  {r['_label']:<34}"
            f" {deferred:>9,}"
            f" {rescued:>9,}"
            f" {expired:>9,}"
            f" {pending:>9,}"
            f" {rate:>7.1%}"
            f"  {r.get('elapsed_s', 0):>7.1f}"
        )
    print()

    # ── Starvation delta ──────────────────────────────────────────────────
    no_starv   = next((r for r in results if "sw=0.0" in r["_label"]), None)
    with_starv = next((r for r in results if "sw=0.5" in r["_label"]), None)
    if no_starv and with_starv:
        print(f"\n  ── Δ (sw=0.0 minus sw=0.5) — isolates starvation term's contribution ──")
        delta_keys = [
            ("JFI (tasks)", "float"), ("JFI (idle)", "float"),
            ("Gini (idle)", "float"), ("CV (idle)", "float"),
            ("Wait (m)", "float"), ("Backlog", "int"),
        ]
        for key, kind in delta_keys:
            d = no_starv.get(key, 0) - with_starv.get(key, 0)
            sign = "+" if d >= 0 else ""
            if kind == "float":
                print(f"    {key:<22}  {sign}{d:.4f}")
            else:
                print(f"    {key:<22}  {sign}{int(d)}")


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
    print("  Starvation Ablation Test")
    print(f"  Composite base: fw=1.4  uw=1.0  k=15  γ=0.1  soft_threshold=0.05")
    print(f"  Testing sw=0.0 (no starvation) vs sw=0.5 (current)")
    print("=" * 72)

    all_csv_rows = []
    all_tables   = []

    # ── Gowalla first (faster) ────────────────────────────────────────────
    if run_gowalla:
        print(f"\n{'━' * 72}")
        print("  GOWALLA  Austin Sep 2010  (compressed, 1:5)")
        print(f"{'━' * 72}")
        w_t, t_t = load_workers_tasks("gowalla", root_path=GOWALLA_ROOT, **GOWALLA_KWARGS)
        print(f"  {len(w_t):,} workers | {len(t_t):,} tasks\n")

        gowalla_res = []
        for label, key, params in STRATEGIES:
            m = run_strategy(w_t, t_t, label, key, params, args.timeout)
            if m is None:
                continue
            m["_label"] = label
            m["_dataset"] = "Gowalla"
            gowalla_res.append(m)
            all_csv_rows.append(m)
        all_tables.append(("GOWALLA  Austin Sep 2010  (compressed, 1:5)", gowalla_res))

    # ── Didi ──────────────────────────────────────────────────────────────
    if run_didi:
        day_path = os.path.join(DIDI_ROOT, DIDI_DAY)
        print(f"\n{'━' * 72}")
        print(f"  DIDI  {DIDI_DAY}")
        print(f"{'━' * 72}")
        w_t, t_t = load_workers_tasks("didi", root_path=day_path)
        print(f"  {len(w_t):,} workers | {len(t_t):,} tasks\n")

        didi_res = []
        for label, key, params in STRATEGIES:
            m = run_strategy(w_t, t_t, label, key, params, args.timeout)
            if m is None:
                continue
            m["_label"] = label
            m["_dataset"] = "Didi"
            didi_res.append(m)
            all_csv_rows.append(m)
        all_tables.append((f"DIDI  {DIDI_DAY}", didi_res))

    # ── Summary ───────────────────────────────────────────────────────────
    for dataset_label, results in all_tables:
        print_results(dataset_label, results)

    # ── CSV ───────────────────────────────────────────────────────────────
    if all_csv_rows:
        csv_path = os.path.join(RESULTS_DIR, f"starvation_ablation_{ts}.csv")
        fieldnames = ["_dataset", "_label", "TAR", "Rev. (k$)",
                      "JFI (tasks)", "JFI (earnings)", "JFI rate", "P10 tasks",
                      "JFI (idle)", "Gini (idle)", "CV (idle)", "CV (earn)",
                      "Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
                      "Backlog", "elapsed_s"]
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_csv_rows)
        print(f"\n  CSV → {csv_path}")


if __name__ == "__main__":
    main()
