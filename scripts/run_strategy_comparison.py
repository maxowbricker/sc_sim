#!/usr/bin/env python3
"""
Full strategy comparison across all registered baselines.

Runs every strategy once per day (default: first available day) in order of
ascending computational cost and prints a detailed metric table to the terminal.

Usage:
    python scripts/run_strategy_comparison.py                  # first available day
    python scripts/run_strategy_comparison.py --days 3         # first 3 days, averaged
    python scripts/run_strategy_comparison.py --day 20161109   # specific day suffix
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

# ---------------------------------------------------------------------------
# Strategy definitions: (display_name, strategy_key, extra_params)
# Ordered from least to most computationally expensive.
# ---------------------------------------------------------------------------
STRATEGIES: List[tuple] = [
    # O(W) immediate scan — simplest possible baseline
    ("Greedy",                "greedy",              {}),
    # O(k) random selection from k-NN
    ("Random",                "random_assign",       {"k": 15}),
    # O(k) EWMA fairness only
    ("EWMA-Only",             "ewma_only",           {"gamma": 0.2}),
    # O(n) deferred cost-balance trigger
    ("Cost-Balancing",        "cost_balancing",      {"alpha": 0.5, "k": 10}),
    # O(W) KVV random-rank scan
    ("BiRanking (BRK)",       "biranking",           {"seed": 42}),
    # O(k) composite with EWMA + starvation + utility
    ("Composite (static)",    "composite",           {
        "fairness_weight": 1.0, "starvation_weight": 0.2,
        "utility_weight": 1.0, "gamma": 0.1, "k": 15,
        "soft_threshold": 0.05,
    }),
    # O(1) roll + O(W) per chosen heuristic
    ("TSGF Sampling",         "tsgf",                {"alpha": 0.4, "beta": 0.3, "gamma": 0.3, "k": 15, "seed": 42}),
    # O(k) + fairness cap tracking
    ("FATP-ANN",              "fatp_ann",            {"mu": 0.5, "alpha_scale": 0.5, "use_k_nearest": True, "k": 15}),
    # O(W*T) Hungarian at fixed review intervals
    ("Discrete Review LP",    "discrete_review_lp",  {"review_period_seconds": 60.0}),
    # O(W) threshold scan — randomised theta
    ("ONRTA-RT",              "onrta_rt",            {"seed": 42}),
    # O(W*T) Hungarian in Stage 2 (triggered at midpoint)
    ("ONRTA-OP",              "onrta_op",            {}),
    # O(W*T) Hungarian on every event — most expensive
    ("MMD-Batch",             "mmd_batch",           {}),
]

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(stats: Dict[str, Any], workers) -> Dict[str, Any]:
    completed   = stats.get("completed_tasks", 0)
    total       = stats.get("total_tasks", 1)
    wait_times  = stats.get("wait_times", [])
    idle_times  = [w.total_idle_time / 60.0 for w in workers]

    # TAR
    tar = completed / total if total else 0.0

    # Revenue
    revenue = stats.get("total_platform_revenue", 0.0)

    # Fairness
    jfi         = stats.get("final_jains_fairness_index", 0.0)
    jfi_earn    = stats.get("final_jfi_earnings", 0.0)

    # JFI rate: fraction of workers who received at least one task
    worker_task_counts = [w.completed_tasks for w in workers]
    jfi_rate = sum(1 for c in worker_task_counts if c > 0) / max(len(worker_task_counts), 1)

    # Wait time
    avg_wait  = float(np.mean(wait_times))  if wait_times else 0.0
    p50_wait  = float(np.percentile(wait_times, 50))  if wait_times else 0.0
    p90_wait  = float(np.percentile(wait_times, 90))  if wait_times else 0.0
    p95_wait  = float(np.percentile(wait_times, 95))  if wait_times else 0.0
    max_wait  = float(np.max(wait_times))   if wait_times else 0.0

    # Worker idle / utilisation
    mean_idle  = float(np.mean(idle_times))  if idle_times else 0.0
    total_shift_min = sum(
        (w.deadline - w.release_time) / 60.0 for w in workers
    )
    total_idle_min  = sum(idle_times)
    utilisation_pct = (
        100.0 * (1.0 - total_idle_min / total_shift_min)
        if total_shift_min > 0 else 0.0
    )

    # Empty km
    empty_km        = stats.get("empty_km", 0.0)
    avg_pickup_km   = stats.get("avg_pickup_distance_km", 0.0)

    # Backlog
    backlog_peak    = stats.get("backlog_peak", 0)

    return {
        "Completed":        completed,
        "Total":            total,
        "TAR":              tar,
        "Revenue ($)":      revenue,
        "JFI (tasks)":      jfi,
        "JFI (earnings)":   jfi_earn,
        "JFI rate":         jfi_rate,
        "Avg Wait (m)":     avg_wait,
        "P50 Wait (m)":     p50_wait,
        "P90 Wait (m)":     p90_wait,
        "P95 Wait (m)":     p95_wait,
        "Max Wait (m)":     max_wait,
        "Mean Idle (m)":    mean_idle,
        "Utilisation (%)":  utilisation_pct,
        "Empty km":         empty_km,
        "Avg Pickup (km)":  avg_pickup_km,
        "Peak Backlog":     backlog_peak,
    }


def _fmt(v: Any, key: str) -> str:
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        if "%" in key:
            return f"{v:.1f}"
        if key in ("Revenue ($)", "Empty km"):
            return f"{v:,.1f}"
        return f"{v:.3f}"
    return str(v)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "Completed", "TAR", "Revenue ($)",
    "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P50 Wait (m)", "P90 Wait (m)", "P95 Wait (m)", "Max Wait (m)",
    "Mean Idle (m)", "Utilisation (%)",
    "Empty km", "Avg Pickup (km)",
    "Peak Backlog",
]

COL_W   = 22   # strategy name column
STAT_W  = 13   # metric value column


def _header(keys: List[str]) -> str:
    row = f"{'Strategy':<{COL_W}}"
    for k in keys:
        row += f" {k:>{STAT_W}}"
    return row


def _divider(keys: List[str]) -> str:
    return "-" * (COL_W + (STAT_W + 1) * len(keys))


def _row(name: str, metrics: Dict[str, Any], keys: List[str]) -> str:
    row = f"{name:<{COL_W}}"
    for k in keys:
        v = metrics.get(k, 0)
        row += f" {_fmt(v, k):>{STAT_W}}"
    return row


def print_table(all_metrics: List[Dict]) -> None:
    keys = METRIC_KEYS

    # Print in two passes to avoid 200-char lines
    block1 = ["Completed", "TAR", "Revenue ($)", "JFI (tasks)", "JFI (earnings)", "JFI rate"]
    block2 = ["Avg Wait (m)", "P50 Wait (m)", "P90 Wait (m)", "P95 Wait (m)", "Max Wait (m)"]
    block3 = ["Mean Idle (m)", "Utilisation (%)", "Empty km", "Avg Pickup (km)", "Peak Backlog"]

    for label, block in [("ASSIGNMENT & FAIRNESS", block1),
                          ("WAIT TIME DISTRIBUTION", block2),
                          ("WORKER UTILISATION & SPATIAL COSTS", block3)]:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        print(_header(block))
        print(_divider(block))
        for row in all_metrics:
            print(_row(row["_name"], row, block))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one_day(day_folder: str, verbose: bool = True) -> List[Dict]:
    day_path = os.path.join(DATA_ROOT, day_folder)
    if verbose:
        print(f"\n  Loading data: {day_folder}")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    n_w, n_t = len(workers_template), len(tasks_template)
    if verbose:
        print(f"  {n_w:,} workers | {n_t:,} tasks")

    results = []
    for display_name, strategy_key, extra_params in STRATEGIES:
        cfg = create_composite_config(
            assignment_strategy=strategy_key,
            **extra_params,
        )

        t0 = time.time()
        try:
            from simulator.simulation import EventSimulator
            sim = EventSimulator(workers_template, tasks_template, cfg)
            sim.reset()
            sim.step()
            stats2 = sim.get_final_results()
            final_workers = list(sim.state.all_workers_map.values())
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  {display_name:<22}  FAILED [{elapsed:.1f}s]  {exc}")
            results.append({"_name": display_name, "_elapsed": elapsed, "_failed": True})
            continue

        elapsed = time.time() - t0

        m = extract_metrics(stats2, final_workers)
        m["_name"]    = display_name
        m["_elapsed"] = elapsed
        m["_failed"]  = False
        results.append(m)

        print(
            f"  {display_name:<22} "
            f"  completed={m['Completed']:>5}/{m['Total']:>5}"
            f"  TAR={m['TAR']:.3f}"
            f"  JFI={m['JFI (tasks)']:.4f}"
            f"  wait={m['Avg Wait (m)']:.2f}m"
            f"  util={m['Utilisation (%)']:.1f}%"
            f"  [{elapsed:.1f}s]"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare all strategies across DiDi days")
    parser.add_argument("--days",   type=int, default=1,
                        help="Number of days to average over (default 1)")
    parser.add_argument("--day",    type=str, default=None,
                        help="Specific day suffix, e.g. 20161109")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV output path")
    args = parser.parse_args()

    # Resolve day folders
    all_days = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    if not all_days:
        print(f"ERROR: No day folders found in {DATA_ROOT}")
        sys.exit(1)

    if args.day:
        match = [d for d in all_days if args.day in d]
        if not match:
            print(f"ERROR: No day folder matching '{args.day}'")
            sys.exit(1)
        eval_days = match[:1]
    else:
        eval_days = all_days[:args.days]

    print("=" * 80)
    print("  STRATEGY BENCHMARK — Spatial Crowdsourcing Simulator")
    print(f"  Days: {', '.join(eval_days)}")
    strat_names = [s[0] for s in STRATEGIES]
    print(f"  Strategies ({len(STRATEGIES)}): " + " | ".join(strat_names))
    print("=" * 80)

    # Accumulate across days
    day_results: List[List[Dict]] = []
    for day in eval_days:
        print(f"\n{'─'*80}")
        print(f"  Day: {day}")
        print(f"{'─'*80}")
        day_results.append(run_one_day(day))

    # Average across days if multiple
    if len(day_results) == 1:
        aggregated = day_results[0]
    else:
        # Build per-strategy averages
        strategy_names = [r["_name"] for r in day_results[0]]
        aggregated = []
        for i, name in enumerate(strategy_names):
            rows = [dr[i] for dr in day_results if i < len(dr)]
            avg_row: Dict[str, Any] = {"_name": name, "_elapsed": sum(r["_elapsed"] for r in rows)}
            for k in METRIC_KEYS:
                vals = [r[k] for r in rows if not r.get("_failed") and k in r]
                avg_row[k] = float(np.mean(vals)) if vals else 0.0
            aggregated.append(avg_row)

    # Print tables
    days_label = f"{len(eval_days)}-day average" if len(eval_days) > 1 else eval_days[0]
    print(f"\n\n{'='*80}")
    print(f"  FINAL RESULTS  ({days_label})")
    print_table(aggregated)

    # Totals row timing
    print(f"\n{'─'*80}")
    print(f"{'Strategy':<22}  {'Wall time':>12}")
    print(f"{'─'*80}")
    total_time = 0.0
    for row in aggregated:
        t = row.get("_elapsed", 0.0)
        total_time += t
        status = "FAILED" if row.get("_failed") else f"{t:.1f}s"
        print(f"  {row['_name']:<22}  {status:>10}")
    print(f"  {'TOTAL':<22}  {total_time:.1f}s")
    print(f"{'='*80}\n")

    # Optional CSV export
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Strategy"] + METRIC_KEYS + ["Wall time (s)"])
            writer.writeheader()
            for row in aggregated:
                out = {"Strategy": row["_name"], "Wall time (s)": round(row.get("_elapsed", 0), 2)}
                for k in METRIC_KEYS:
                    out[k] = row.get(k, 0)
                writer.writerow(out)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
