#!/usr/bin/env python3
"""
NYC Taxi — Worker Instantiation Sweep

Tests different worker bootstrapping configurations against Greedy and Composite
strategies on a single NYC day, to find the most realistic worker count before
running the full strategy comparison.

The "representative Wednesday" (2012-05-09) is used as the comparable NYC
equivalent of the Didi benchmark day (20161109, also a Wednesday).

Worker configs tested:
  - Fixed fleet sizes: 2k, 5k, 10k workers
  - Proportional to task count: 1:4, 1:5, 1:7 task-to-worker ratios

Because a full NYC day (~490k trips) is significantly larger than a Didi day,
use --sample N to apply stratified sampling down to N tasks for a faster sweep
that is more directly comparable to Didi volumes.

Usage:
    python scripts/run_nyc_worker_sweep.py
    python scripts/run_nyc_worker_sweep.py --date 2012-05-09
    python scripts/run_nyc_worker_sweep.py --sample 50000          # trim to 50k tasks
    python scripts/run_nyc_worker_sweep.py --output nyc_sweep.csv
"""

from __future__ import annotations

import argparse
import copy
import csv as _csv
import os
import sys
import threading
import time
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config, get_data_sampling_config
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NYC_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "nyc_taxi")

# Representative Wednesday in May 2012 — comparable to Didi 20161109 (also Wed)
DEFAULT_DATE = "2012-05-09"

# Strategies to test (just the two of interest for this sweep)
STRATEGIES: List[tuple] = [
    ("Greedy",             "greedy",   {}),
    ("Composite (static)", "composite", {
        "fairness_weight": 1.0,
        "starvation_weight": 0.2,
        "utility_weight": 1.0,
        "gamma": 0.1,
        "k": 15,
        "soft_threshold": 0.05,
    }),
]

# Worker configurations to sweep
# Each entry: (label, use_proportional, workers_per_task_ratio, num_workers)
# Fixed counts are calibrated for NYC's ~780km² footprint across 265 zones
# (much larger than Didi's ~30km² Chengdu area, which used 10k-20k workers).
WORKER_CONFIGS: List[tuple] = [
    ("Fixed-10000",  False, None,  10000),
    ("Fixed-20000",  False, None,  20000),
    ("Fixed-40000",  False, None,  40000),
    ("Prop-1:7",     True,  1/7,   None),
    ("Prop-1:5",     True,  1/5,   None),
    ("Prop-1:4",     True,  1/4,   None),
]

# ---------------------------------------------------------------------------
# Metric extraction (mirrors run_strategy_comparison.py)
# ---------------------------------------------------------------------------

def extract_metrics(stats: Dict[str, Any], workers) -> Dict[str, Any]:
    completed  = stats.get("completed_tasks", 0)
    total      = stats.get("total_tasks", 1)
    wait_times = stats.get("wait_times", [])
    idle_times = [w.total_idle_time / 60.0 for w in workers]

    tar       = completed / total if total else 0.0
    revenue   = stats.get("total_platform_revenue", 0.0)
    jfi       = stats.get("final_jains_fairness_index", 0.0)
    jfi_earn  = stats.get("final_jfi_earnings", 0.0)

    worker_task_counts = [w.completed_tasks for w in workers]
    jfi_rate = sum(1 for c in worker_task_counts if c > 0) / max(len(worker_task_counts), 1)

    avg_wait = float(np.mean(wait_times))              if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95))    if wait_times else 0.0
    max_wait = float(np.max(wait_times))               if wait_times else 0.0

    total_shift_min = sum((w.deadline - w.release_time) / 60.0 for w in workers)
    total_idle_min  = sum(idle_times)
    utilisation_pct = (
        max(0.0, min(100.0, 100.0 * (1.0 - total_idle_min / total_shift_min)))
        if total_shift_min > 0 else 0.0
    )

    return {
        "Workers":          len(workers),
        "Tasks":            total,
        "Completed":        completed,
        "TAR":              tar,
        "Revenue ($)":      revenue,
        "JFI (tasks)":      jfi,
        "JFI (earnings)":   jfi_earn,
        "JFI rate":         jfi_rate,
        "Avg Wait (m)":     avg_wait,
        "P95 Wait (m)":     p95_wait,
        "Max Wait (m)":     max_wait,
        "Utilisation (%)":  utilisation_pct,
        "Avg Pickup (km)":  stats.get("avg_pickup_distance_km", 0.0),
        "Peak Backlog":     stats.get("backlog_peak", 0),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "Workers", "Tasks", "Completed", "TAR",
    "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P95 Wait (m)",
    "Utilisation (%)", "Avg Pickup (km)", "Peak Backlog",
]

COL_W  = 30   # worker config + strategy column
STAT_W = 13


def _fmt(v: Any, key: str) -> str:
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        if "%" in key:
            return f"{v:.1f}"
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _header(keys: List[str]) -> str:
    return f"{'Config / Strategy':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in keys)


def _divider(keys: List[str]) -> str:
    return "-" * (COL_W + (STAT_W + 1) * len(keys))


def _row(name: str, metrics: Dict[str, Any], keys: List[str]) -> str:
    return f"{name:<{COL_W}}" + "".join(f" {_fmt(metrics.get(k, 0), k):>{STAT_W}}" for k in keys)


def print_table(all_results: List[Dict]) -> None:
    block1 = ["Workers", "Tasks", "Completed", "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"]
    block2 = ["Avg Wait (m)", "P95 Wait (m)", "Utilisation (%)", "Avg Pickup (km)", "Peak Backlog"]

    for label, block in [("ASSIGNMENT & FAIRNESS", block1), ("WAIT / UTILISATION / SPATIAL", block2)]:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        print(_header(block))
        print(_divider(block))
        prev_wconfig = None
        for row in all_results:
            if row["_wconfig"] != prev_wconfig:
                if prev_wconfig is not None:
                    print()
                prev_wconfig = row["_wconfig"]
            print(_row(row["_name"], row, block))


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_sim_in_thread(sim, cancel_event: threading.Event, exc_holder: dict):
    try:
        sim.step(cancel_event=cancel_event)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_combination(
    workers_template,
    tasks_template,
    display_name: str,
    strategy_key: str,
    extra_params: dict,
    timeout_sec: float = 2700.0,
) -> Dict[str, Any]:
    """Run one (worker_config × strategy) combination and return a metric dict."""
    cfg = create_composite_config(assignment_strategy=strategy_key, **extra_params)

    t0 = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator

        sim = EventSimulator(workers_template, tasks_template, cfg)
        sim.reset()

        cancel_event = threading.Event()
        thread = threading.Thread(
            target=_run_sim_in_thread, args=(sim, cancel_event, exc_holder), daemon=True
        )
        thread.start()
        thread.join(timeout=timeout_sec)

        timed_out = thread.is_alive()
        if timed_out:
            cancel_event.set()
            thread.join(timeout=60)

        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    {display_name:<28}  FAILED [{elapsed:.1f}s]  {exc}")
        return {"_name": display_name, "_failed": True, "_elapsed": elapsed}

    elapsed = time.time() - t0
    tag = f"TIMEOUT (partial) [{elapsed:.0f}s]" if timed_out else f"[{elapsed:.1f}s]"
    m = extract_metrics(stats, workers)
    print(
        f"    {display_name:<28}"
        f"  workers={m['Workers']:>6,}  tasks={m['Tasks']:>6,}"
        f"  TAR={m['TAR']:.3f}  JFI={m['JFI (tasks)']:.4f}"
        f"  wait={m['Avg Wait (m)']:.2f}m  util={m['Utilisation (%)']:.1f}%"
        f"  {tag}"
    )
    m["_name"]    = display_name
    m["_failed"]  = False
    m["_elapsed"] = elapsed
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date",    type=str, default=DEFAULT_DATE,
                        help=f"ISO date to load (default: {DEFAULT_DATE})")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Apply stratified sampling to N tasks after loading "
                             "(e.g. 50000 to match typical Didi volumes)")
    parser.add_argument("--output",  type=str, default=None,
                        help="Path to write results CSV (default: nyc_worker_sweep_<date>.csv)")
    parser.add_argument("--timeout", type=float, default=2700.0,
                        help="Per-run timeout in seconds (default: 2700 = 45 min)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        PROJECT_ROOT, f"nyc_worker_sweep_{args.date.replace('-', '')}.csv"
    )

    print("=" * 80)
    print("  NYC TAXI — Worker Instantiation Sweep")
    print(f"  Date:     {args.date}  (representative Wednesday, comparable to Didi 20161109)")
    print(f"  Sampling: {f'stratified → {args.sample:,} tasks' if args.sample else 'full day (no sampling)'}")
    print(f"  Strategies: {', '.join(s[0] for s in STRATEGIES)}")
    print(f"  Worker configs ({len(WORKER_CONFIGS)}): {', '.join(c[0] for c in WORKER_CONFIGS)}")
    print(f"  Output: {output_path}")
    print("=" * 80)

    all_results: List[Dict] = []

    for wlabel, use_prop, ratio, n_fixed in WORKER_CONFIGS:
        print(f"\n{'─'*80}")
        if use_prop:
            print(f"  Worker config: {wlabel}  (proportional, ratio={ratio:.4f})")
        else:
            print(f"  Worker config: {wlabel}  (fixed {n_fixed:,} workers)")
        print(f"{'─'*80}")

        # Build adapter kwargs for this worker config
        adapter_kwargs: Dict[str, Any] = {
            "date": args.date,
            "use_proportional_workers": use_prop,
            "random_state": 42,
        }
        if use_prop:
            adapter_kwargs["workers_per_task_ratio"] = ratio
        else:
            adapter_kwargs["use_proportional_workers"] = False
            adapter_kwargs["num_workers"] = n_fixed

        print(f"  Loading NYC data for {args.date}...")
        t_load = time.time()
        workers_raw, tasks_raw = load_workers_tasks(
            "nyc_taxi", root_path=NYC_DATA_ROOT, **adapter_kwargs
        )
        print(f"  Loaded: {len(workers_raw):,} workers | {len(tasks_raw):,} tasks  ({time.time()-t_load:.1f}s)")

        # Optional stratified sampling to bring task volume down to a comparable scale
        if args.sample and len(tasks_raw) > args.sample:
            sampling_cfg = get_data_sampling_config()

            # For proportional configs the worker count must be re-derived from the
            # *sampled* task count, otherwise the full-file ratio (e.g. 69k workers
            # for 483k tasks) is preserved while tasks shrink to 50k, inverting the
            # supply:demand ratio.  Fixed configs keep their explicit count as-is.
            if use_prop and ratio is not None:
                target_w = max(1, round(args.sample * ratio))
            else:
                target_w = len(workers_raw)  # fixed count — keep exactly what was loaded

            target_w = min(target_w, len(workers_raw))

            tasks_s, w_dict = stratified_temporal_sample(
                all_workers=workers_raw,
                all_tasks=tasks_raw,
                target_tasks=args.sample,
                worker_counts=[target_w],
                num_bins=sampling_cfg.get("stratified_sampling_bins", 288),
                seed=sampling_cfg.get("random_state", 42),
            )
            workers_template = w_dict[target_w]
            tasks_template   = tasks_s
            print(f"  After sampling: {len(workers_template):,} workers | {len(tasks_template):,} tasks  "
                  f"(ratio 1:{len(tasks_template)//max(len(workers_template),1)})")
        else:
            workers_template = workers_raw
            tasks_template   = tasks_raw

        # Run each strategy against this worker config
        for strat_display, strat_key, strat_params in STRATEGIES:
            run_label = f"{wlabel} / {strat_display}"
            m = run_combination(
                workers_template,
                tasks_template,
                run_label,
                strat_key,
                strat_params,
                timeout_sec=args.timeout,
            )
            m["_wconfig"]   = wlabel
            m["_strategy"]  = strat_display
            m["_date"]      = args.date
            m["_sampled"]   = args.sample or len(tasks_raw)
            all_results.append(m)

    # Print summary table
    successful = [r for r in all_results if not r.get("_failed")]
    if successful:
        print(f"\n\n{'='*80}")
        print(f"  FINAL RESULTS — NYC Worker Sweep  ({args.date})")
        print_table(successful)
    else:
        print("\n⚠️  All runs failed — check error messages above.")

    # Timing summary
    print(f"\n{'─'*80}")
    print(f"{'Config / Strategy':<30}  {'Wall time':>12}")
    print(f"{'─'*80}")
    total_time = 0.0
    prev_wc = None
    for row in all_results:
        if row["_wconfig"] != prev_wc:
            if prev_wc is not None:
                print()
            prev_wc = row["_wconfig"]
        t = row.get("_elapsed", 0.0)
        total_time += t
        status = "FAILED" if row.get("_failed") else f"{t:.1f}s"
        print(f"  {row['_name']:<30}  {status:>12}")
    print(f"  {'TOTAL':<30}  {total_time:.1f}s")

    # Save CSV
    csv_fields = ["Date", "Sampled Tasks", "Worker Config", "Strategy"] + METRIC_KEYS + ["Wall time (s)", "Status"]
    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow({
                "Date":            row.get("_date", args.date),
                "Sampled Tasks":   row.get("_sampled", ""),
                "Worker Config":   row.get("_wconfig", ""),
                "Strategy":        row.get("_strategy", ""),
                **{k: (round(row[k], 6) if isinstance(row.get(k), float) else row.get(k, ""))
                   for k in METRIC_KEYS},
                "Wall time (s)":   round(row.get("_elapsed", 0), 2),
                "Status":          "FAILED" if row.get("_failed") else "OK",
            })
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
