#!/usr/bin/env python3
"""
Parameter Tuning — Aveklouris LP (Discrete Review) review_period Sweep

The review_period_seconds hyperparameter controls how long the Discrete Review
LP baseline buffers arrivals before running the Hungarian assignment. It must be
tuned per dataset:

  - Too short  → near-greedy behaviour (very small batches, no thickness benefit)
  - Too long   → tasks/workers expire before the next review epoch fires (TAR drops)

This script sweeps review_period across a range on BOTH Didi and Gowalla, reporting
TAR, JFI, average wait, peak backlog, and runtime for each configuration.

The selected review_period for the final benchmark is the value that maximises
TAR on each dataset (tie-broken by lowest wait time).

Output:
    results/parameter_tuning/review_period_didi.csv
    results/parameter_tuning/review_period_gowalla.csv

Usage:
    python scripts/experiments/parameter_tuning/sweep_review_period.py
    python scripts/experiments/parameter_tuning/sweep_review_period.py --didi-only
    python scripts/experiments/parameter_tuning/sweep_review_period.py --gowalla-only
    python scripts/experiments/parameter_tuning/sweep_review_period.py --timeout 600
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

# ---------------------------------------------------------------------------
# Review period values to test (seconds)
# ---------------------------------------------------------------------------
# Gowalla tasks expire after exactly 1800s; periods beyond ~600s risk heavy expiry.
# Didi task expiry windows are 15-45 min, so larger periods are safer there.

PERIODS_DIDI    = [5, 10, 15, 30, 60]   # seconds
PERIODS_GOWALLA = [5, 10, 15, 30, 60]         # capped at 300s (30-min expiry)

# ---------------------------------------------------------------------------
# Dataset paths / loaders
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
    workers_per_task_ratio = 0.20,   # 1:5 ratio
    random_state           = 42,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "parameter_tuning")

DEFAULT_TIMEOUT = 900   # 15 min per run — LP can be slow on Didi at long periods

FIELDNAMES = [
    "dataset", "review_period_s",
    "TAR", "JFI (tasks)", "JFI (earnings)",
    "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
    "Peak Backlog", "Completed", "Total", "elapsed_s",
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

    avg_wait = float(np.mean(wait_times))           if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95)) if wait_times else 0.0

    return {
        "TAR":             tar,
        "JFI (tasks)":     jfi,
        "JFI (earnings)":  jfi_earn,
        "Avg Wait (m)":    avg_wait,
        "P95 Wait (m)":    p95_wait,
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
        "Peak Backlog":    stats.get("backlog_peak", 0),
        "Completed":       completed,
        "Total":           total,
    }


def _sim_thread(sim, exc_holder: dict) -> None:
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_one(workers_tmpl, tasks_tmpl, period_s: float,
            timeout_sec: float) -> Optional[Dict[str, Any]]:
    cfg = create_composite_config(
        assignment_strategy="discrete_review_lp",
        review_period_seconds=period_s,
    )
    exc_holder: dict = {}
    t0 = time.time()

    try:
        from simulator.simulation import EventSimulator
        sim = EventSimulator(
            copy.deepcopy(workers_tmpl),
            copy.deepcopy(tasks_tmpl),
            cfg,
        )
        sim.reset()
        thread = threading.Thread(target=_sim_thread, args=(sim, exc_holder), daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            print("TIMEOUT")
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
COL_W  = 14
STAT_W = 10

DISPLAY_KEYS = ["TAR", "JFI (tasks)", "Avg Wait (m)", "P95 Wait (m)", "Peak Backlog"]


def _fmt(v, key: str) -> str:
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)"):
            return f"{v:.4f}"
        if key in ("Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)"):
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)


def _header() -> str:
    return (
        f"{'Period (s)':<{COL_W}}"
        + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS)
        + f"  {'Time (s)':>8}"
    )


def _sep() -> str:
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 10)


def _row(period_s: float, m: Dict, best_tar: float) -> str:
    marker = "  ← best TAR" if abs(m["TAR"] - best_tar) < 1e-6 else ""
    label  = f"{int(period_s)}s{marker}"
    cells  = "".join(f" {_fmt(m.get(k, 0), k):>{STAT_W}}" for k in DISPLAY_KEYS)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>8.1f}"


def _recommendation(results: List[Dict], periods: List[float]) -> float:
    """Return the period with the highest TAR (tie-broken by lowest wait)."""
    best = max(results, key=lambda r: (r["TAR"], -r["Avg Wait (m)"]))
    return periods[results.index(best)]


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(
    dataset_name: str,
    workers_tmpl,
    tasks_tmpl,
    periods: List[float],
    timeout_sec: float,
    writer: _csv.DictWriter,
    f,          # open file handle for flush
) -> List[Dict]:
    results: List[Dict] = []

    for period in periods:
        label = f"period={int(period)}s"
        print(f"  {label:<18}", end="  ", flush=True)

        m = run_one(workers_tmpl, tasks_tmpl, period, timeout_sec)
        if m is None:
            continue

        print(
            f"TAR={m['TAR']:.4f}  "
            f"JFI={m['JFI (tasks)']:.4f}  "
            f"wait={m['Avg Wait (m)']:.2f}m  "
            f"backlog={m['Peak Backlog']}  "
            f"[{m['elapsed_s']:.1f}s]"
        )
        row = {"dataset": dataset_name, "review_period_s": period, **m}
        writer.writerow(row)
        f.flush()
        m["period_s"] = period
        results.append(m)

    return results


def print_summary(dataset_name: str, results: List[Dict], periods: List[float]) -> None:
    if not results:
        print("  No results to display.")
        return

    best_tar = max(r["TAR"] for r in results)

    print(f"\n\n{'=' * 70}")
    print(f"  {dataset_name} — Discrete Review LP: review_period sweep")
    print(f"{'=' * 70}\n")
    print(_sep())
    print(_header())
    print(_sep())
    for r in results:
        print(_row(r["period_s"], r, best_tar))
    print(_sep())

    rec_period = _recommendation(results, [r["period_s"] for r in results])
    print(f"\n  Recommended review_period for {dataset_name}: {int(rec_period)}s")

    tars  = [r["TAR"]          for r in results]
    waits = [r["Avg Wait (m)"] for r in results]
    print(f"  TAR range:  {min(tars):.4f} – {max(tars):.4f}  (Δ {max(tars)-min(tars):.4f})")
    print(f"  Wait range: {min(waits):.2f} – {max(waits):.2f} min")


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
    parser.add_argument("--timeout",      type=float, default=DEFAULT_TIMEOUT,
                        help="Per-run timeout in seconds (default: 900)")
    args = parser.parse_args()

    run_didi    = not args.gowalla_only
    run_gowalla = not args.didi_only

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  Parameter Tuning — Aveklouris LP review_period sweep")
    print(f"  Timeout per run: {args.timeout:.0f}s")
    print("=" * 70)

    # ── Gowalla ──────────────────────────────────────────────────────────────
    if run_gowalla:
        gowalla_csv = os.path.join(RESULTS_DIR, "review_period_gowalla.csv")
        print(f"\n{'─' * 70}")
        print(f"  Dataset: Gowalla Austin Sep 2010 (compressed, 1:5 ratio)")
        print(f"  Periods: {PERIODS_GOWALLA} s")
        print(f"  Note: task expiry = 1800s — periods >300s risk heavy expiry losses")
        print(f"  Output: {gowalla_csv}")
        print(f"{'─' * 70}\n")

        print("  Loading Gowalla ...", end="  ", flush=True)
        workers_g, tasks_g = load_workers_tasks("gowalla", root_path=GOWALLA_ROOT,
                                                **GOWALLA_KWARGS)
        print(f"{len(workers_g):,} workers | {len(tasks_g):,} tasks\n")

        with open(gowalla_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            g_results = run_dataset(
                "Gowalla", workers_g, tasks_g,
                PERIODS_GOWALLA, args.timeout, writer, f,
            )

        print_summary("Gowalla", g_results, PERIODS_GOWALLA)
        print(f"\n  Results saved → {gowalla_csv}")

    # ── Didi ─────────────────────────────────────────────────────────────────
    if run_didi:
        didi_csv = os.path.join(RESULTS_DIR, "review_period_didi.csv")
        day_path = os.path.join(DIDI_ROOT, DIDI_DAY)
        print(f"\n{'─' * 70}")
        print(f"  Dataset: Didi Chuxing Chengdu 20161109")
        print(f"  Periods: {PERIODS_DIDI} s")
        print(f"  Note: ~2.5 tasks/sec; long periods build large batches (slower LP)")
        print(f"  Output: {didi_csv}")
        print(f"{'─' * 70}\n")

        print(f"  Loading {DIDI_DAY} ...", end="  ", flush=True)
        workers_d, tasks_d = load_workers_tasks("didi", root_path=day_path)
        print(f"{len(workers_d):,} workers | {len(tasks_d):,} tasks\n")

        with open(didi_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            d_results = run_dataset(
                "Didi", workers_d, tasks_d,
                PERIODS_DIDI, args.timeout, writer, f,
            )

        print_summary("Didi", d_results, PERIODS_DIDI)
        print(f"\n  Results saved → {didi_csv}")

    print(f"\n{'=' * 70}")
    print("  Sweep complete. Update config.py review_period_seconds per dataset")
    print("  with the recommended values above before running the final benchmark.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
