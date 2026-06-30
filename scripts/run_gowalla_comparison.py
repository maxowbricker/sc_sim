#!/usr/bin/env python3
"""
Gowalla LBSN — Strategy Comparison

Runs Greedy and Composite strategies against the Stanford Gowalla check-in
dataset (Austin preset) across different worker:task ratios and — crucially —
with and without temporal day-compression, so you can see the effect directly.

Why temporal compression matters:
    LBSN check-in rates are ~150x lower than ride-hailing trips. Without
    compression, September 2010 Austin has only ~31 tasks active at any
    moment (spread over 29 days), giving strategies nothing to compete over
    and producing trivially identical TAR / JFI across all strategies.
    compress_to_day=True strips the calendar date from every check-in,
    keeping only the wall-clock time (HH:MM:SS), and stacks all events
    onto a single 24-hour reference window. This raises concurrent task
    density from ~31 to ~912, matching Didi's ~2-4 workers per concurrent
    task and making strategy differentiation meaningful.

Sweep dimensions:
    - Compression: yes / no (or both, default)
    - Worker:task ratio: 1:5 (default), 1:7, 1:4
    - Strategy: Greedy, Composite (static)

Data window — "Austin September 2010":
    ~43,788 check-ins over 29 days, 8,758 unique (user, day) workers.
    Excluding March 2010 (74k — dominated by SXSW festival).
    After compression: 24h simulation window, ~912 concurrent tasks,
    ~2,919 concurrent workers (1:5 ratio) — comparable to Didi.

Usage:
    # Full sweep (both compression modes, all ratios, both strategies)
    python scripts/run_gowalla_comparison.py

    # Compressed only — fastest meaningful run
    python scripts/run_gowalla_comparison.py --compression compressed

    # Uncompressed only — shows pathological low-density baseline
    python scripts/run_gowalla_comparison.py --compression uncompressed

    # Fix ratio to 1:5, run both compression modes
    python scripts/run_gowalla_comparison.py --ratio 0.2

    # Different date window (e.g. busier Aug+Sep combined)
    python scripts/run_gowalla_comparison.py --date-start 2010-08-01 --date-end 2010-09-30

    # San Francisco region
    python scripts/run_gowalla_comparison.py --region san_francisco
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

from config import create_composite_config, get_data_sampling_config
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

GOWALLA_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "gowalla")

DEFAULT_REGION     = "austin"
DEFAULT_DATE_START = "2010-09-01"
DEFAULT_DATE_END   = "2010-09-30"

STRATEGIES: List[Tuple[str, str, dict]] = [
    # ── Proposed strategies ───────────────────────────────────────────────────
    ("Greedy",                "greedy",              {}),
    ("k-NLF (k=15)",          "knlf",                {"k": 15}),
    ("Composite (static)",    "composite",           {
        "fairness_weight": 1.6, "starvation_weight": 0.0,
        "utility_weight": 1.0, "gamma": 0.1, "k": 15,
        "soft_threshold": 0.0,
    }),
    # ── O(k) signal variants ──────────────────────────────────────────────────
    ("EWMA-Only",             "ewma_only",           {"gamma": 0.2}),
    ("k-NTF-EPH (k=15)",      "kntf_eph",            {"k": 15}),
    ("k-NTF-IR (k=15)",       "kntf_ir",             {"k": 15}),
    ("Random",                "random_assign",       {"k": 15}),
    # ── O(W) unconstrained baselines ─────────────────────────────────────────
    ("LAF",                   "laf",                 {}),
    ("BiRanking (BRK)",       "biranking",           {"seed": 42}),
    ("FATP-ANN",              "fatp_ann",            {"mu": 1.5, "alpha_scale": 0.5, "use_k_nearest": True, "k": 15}),
    # ── Heavy / batch baselines ───────────────────────────────────────────────
    ("ONRTA-RT",              "onrta_rt",            {"seed": 42}),
    ("ONRTA-OP",              "onrta_op",            {}),
    ("Discrete Review LP",    "discrete_review_lp",  {"review_period_seconds": 15.0}),
]

# (label, workers_per_task_ratio)
RATIO_CONFIGS: List[Tuple[str, float]] = [
    ("Ratio 1:5",  0.20),
    ("Ratio 1:7",  1 / 7),
    ("Ratio 1:4",  0.25),
]

# (label, compress_to_day)
COMPRESS_CONFIGS: List[Tuple[str, bool]] = [
    ("Compressed",   True),
    ("Uncompressed", False),
]


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
    revenue  = stats.get("total_platform_revenue", 0.0)

    worker_task_counts = [w.completed_tasks for w in workers]
    jfi_rate = sum(1 for c in worker_task_counts if c > 0) / max(len(worker_task_counts), 1)

    avg_wait = float(np.mean(wait_times))           if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95)) if wait_times else 0.0

    idle_min       = sum(w.total_idle_time / 60.0 for w in workers)
    shift_min      = sum((w.deadline - w.release_time) / 60.0 for w in workers)
    util_pct = (
        max(0.0, min(100.0, 100.0 * (1.0 - idle_min / shift_min)))
        if shift_min > 0 else 0.0
    )

    return {
        "Workers":         len(workers),
        "Tasks":           total,
        "Completed":       completed,
        "TAR":             tar,
        "Revenue ($)":     revenue,
        "JFI (tasks)":     jfi,
        "JFI (earnings)":  jfi_earn,
        "JFI rate":        jfi_rate,
        "Avg Wait (m)":    avg_wait,
        "P95 Wait (m)":    p95_wait,
        "Utilisation (%)": util_pct,
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
        "Peak Backlog":    stats.get("backlog_peak", 0),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "Workers", "Tasks", "TAR",
    "Revenue ($)",
    "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P95 Wait (m)",
    "Utilisation (%)", "Avg Pickup (km)",
]

COL_W  = 38
STAT_W = 13


def _fmt(v: Any, key: str) -> str:
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        if key == "Revenue ($)":
            return f"{v:,.1f}"
        if "%" in key:
            return f"{v:.1f}"
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _sep(keys: List[str]) -> str:
    return "─" * (COL_W + (STAT_W + 1) * len(keys))


def _header(keys: List[str]) -> str:
    return f"{'Config / Strategy':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in keys)


def _row(name: str, metrics: Dict[str, Any], keys: List[str]) -> str:
    return f"{name:<{COL_W}}" + "".join(f" {_fmt(metrics.get(k, 0), k):>{STAT_W}}" for k in keys)


def print_summary_table(results: List[Dict]) -> None:
    """Print final grouped summary table."""
    print()
    print(_sep(METRIC_KEYS))
    print(_header(METRIC_KEYS))
    print(_sep(METRIC_KEYS))

    prev_compress = None
    prev_ratio    = None
    for r in results:
        if r.get("_failed"):
            continue
        if r["_compress"] != prev_compress:
            if prev_compress is not None:
                print()
            print(f"  [ {'Compressed' if r['_compress'] else 'Uncompressed (baseline)'} ]")
            prev_compress = r["_compress"]
            prev_ratio = None
        if r["_ratio"] != prev_ratio:
            if prev_ratio is not None:
                print()
            prev_ratio = r["_ratio"]
        print(_row(r["_name"], r, METRIC_KEYS))

    print(_sep(METRIC_KEYS))


def save_csv(results: List[Dict], path: str) -> None:
    all_keys = (
        ["_name", "_compress", "_ratio", "_strategy",
         "_region", "_date_start", "_date_end", "_elapsed"]
        + METRIC_KEYS
    )
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in results if not r.get("_failed"))
    print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_sim_in_thread(sim, cancel_event, exc_holder):
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_combination(
    workers_template,
    tasks_template,
    display_name: str,
    strategy_key: str,
    extra_params: dict,
    timeout_sec: float = 1800.0,
) -> Dict[str, Any]:
    cfg = create_composite_config(assignment_strategy=strategy_key, **extra_params)

    t0 = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator

        sim = EventSimulator(
            copy.deepcopy(workers_template),
            copy.deepcopy(tasks_template),
            cfg,
        )
        sim.reset()

        cancel_event = threading.Event()
        thread = threading.Thread(
            target=_run_sim_in_thread, args=(sim, cancel_event, exc_holder), daemon=True,
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
        print(f"    {display_name:<36}  FAILED [{elapsed:.1f}s]  {exc}")
        return {"_name": display_name, "_failed": True, "_elapsed": elapsed}

    elapsed = time.time() - t0
    tag = f"TIMEOUT [{elapsed:.0f}s]" if timed_out else f"[{elapsed:.1f}s]"
    m = extract_metrics(stats, workers)
    print(
        f"    {display_name:<36}"
        f"  w={m['Workers']:>5,}  t={m['Tasks']:>6,}"
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--region", type=str, default=DEFAULT_REGION,
        help=f"Gowalla region preset (default: {DEFAULT_REGION}). "
             "Options: austin, san_francisco",
    )
    parser.add_argument(
        "--date-start", type=str, default=DEFAULT_DATE_START,
        help=f"Start date ISO (default: {DEFAULT_DATE_START})",
    )
    parser.add_argument(
        "--date-end", type=str, default=DEFAULT_DATE_END,
        help=f"End date ISO (default: {DEFAULT_DATE_END})",
    )
    parser.add_argument(
        "--compression", type=str, default="both",
        choices=["both", "compressed", "uncompressed"],
        help=(
            "Which compression modes to run (default: both). "
            "'compressed' = temporal day-stacking (recommended); "
            "'uncompressed' = raw timestamps spread over full date range; "
            "'both' = run each mode back-to-back for direct comparison."
        ),
    )
    parser.add_argument(
        "--ratio", type=float, default=None,
        help="Fix a single worker:task ratio (e.g. 0.2 = 1:5) instead of "
             "sweeping all three. Useful for a quick sanity check.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Stratified-sample to N tasks after loading (e.g. 20000). "
             "Generally not needed — Sep 2010 Austin already fits in memory.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV path for results (default: auto-named in project root)",
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0,
        help="Per-run timeout in seconds (default: 1800 = 30 min)",
    )
    args = parser.parse_args()

    # Resolve which compression modes to run
    if args.compression == "both":
        compress_modes = COMPRESS_CONFIGS
    elif args.compression == "compressed":
        compress_modes = [("Compressed", True)]
    else:
        compress_modes = [("Uncompressed", False)]

    # Resolve which ratios to sweep
    ratio_configs = (
        [(f"Ratio {args.ratio:.2f}", args.ratio)]
        if args.ratio is not None
        else RATIO_CONFIGS
    )

    date_tag = f"{args.date_start}_to_{args.date_end}".replace("-", "")
    output_path = args.output or os.path.join(
        PROJECT_ROOT, f"gowalla_{args.region}_{date_tag}.csv"
    )

    n_runs = len(compress_modes) * len(ratio_configs) * len(STRATEGIES)

    print("=" * 80)
    print("  Gowalla LBSN — Strategy Comparison")
    print(f"  Region:      {args.region}")
    print(f"  Date range:  {args.date_start} → {args.date_end}")
    print(f"  Compression: {args.compression}")
    print(f"  Ratios:      {', '.join(r[0] for r in ratio_configs)}")
    print(f"  Strategies:  {', '.join(s[0] for s in STRATEGIES)}")
    print(f"  Total runs:  {n_runs}")
    print(f"  Output:      {output_path}")
    print("=" * 80)

    all_results: List[Dict] = []

    for compress_label, compress in compress_modes:
        print(f"\n{'━'*80}")
        print(f"  COMPRESSION MODE: {compress_label.upper()}")
        if compress:
            print("  (all timestamps mapped to a single 24h reference window)")
        else:
            print("  (raw timestamps — events spread over the full date range)")
        print(f"{'━'*80}")

        for ratio_label, ratio in ratio_configs:
            print(f"\n  {'─'*76}")
            print(f"  {ratio_label}  (workers = tasks * {ratio:.4f})")
            print(f"  {'─'*76}")

            adapter_kwargs = {
                "region":                 args.region,
                "date_start":             args.date_start,
                "date_end":               args.date_end,
                "task_mode":              "checkin",
                "task_window_hours":      0.5,
                "shift_hours":            8.0,
                "dropoff_noise_km":       2.0,
                "compress_to_day":        compress,
                "workers_per_task_ratio": ratio,
                "random_state":           42,
            }

            t_load = time.time()
            workers_raw, tasks_raw = load_workers_tasks(
                "gowalla", root_path=GOWALLA_DATA_ROOT, **adapter_kwargs
            )
            load_s = time.time() - t_load

            rt = np.array([task.release_time for task in tasks_raw])
            et = np.array([task.expire_time   for task in tasks_raw])
            span_h    = (rt.max() - rt.min()) / 3600 if len(rt) > 1 else 0.0
            avg_win_m = float(np.mean((et - rt) / 60))
            conc_t    = len(tasks_raw)  * (avg_win_m  / (span_h * 60)) if span_h else 0
            conc_w    = len(workers_raw) * (8 / span_h)                  if span_h else 0

            print(
                f"  Loaded: {len(workers_raw):,} workers | {len(tasks_raw):,} tasks  "
                f"({load_s:.1f}s)  span={span_h:.1f}h  "
                f"concurrent: ~{conc_t:.0f} tasks / ~{conc_w:.0f} workers"
            )

            # Optional stratified sampling
            if args.sample and len(tasks_raw) > args.sample:
                sampling_cfg = get_data_sampling_config()
                target_w = min(max(1, round(args.sample * ratio)), len(workers_raw))
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
                print(
                    f"  After sampling: {len(workers_template):,} workers "
                    f"| {len(tasks_template):,} tasks"
                )
            else:
                workers_template = workers_raw
                tasks_template   = tasks_raw

            for strat_display, strat_key, strat_params in STRATEGIES:
                run_label = f"{compress_label[:11]} | {ratio_label[:9]} | {strat_display}"
                m = run_combination(
                    workers_template, tasks_template,
                    run_label, strat_key, strat_params,
                    timeout_sec=args.timeout,
                )
                m["_compress"]   = compress
                m["_ratio"]      = ratio_label
                m["_strategy"]   = strat_display
                m["_region"]     = args.region
                m["_date_start"] = args.date_start
                m["_date_end"]   = args.date_end
                m["_sampled"]    = args.sample or len(tasks_raw)
                all_results.append(m)

    # Final summary
    successful = [r for r in all_results if not r.get("_failed")]
    if successful:
        print(f"\n\n{'='*80}")
        print(f"  FINAL RESULTS — Gowalla {args.region}  "
              f"({args.date_start} → {args.date_end})")
        print_summary_table(successful)
    else:
        print("\n  All runs failed — check error messages above.")

    # Timing summary
    print(f"\n{'─'*50}  {'Wall time':>10}")
    total_time = 0.0
    for row in all_results:
        t = row.get("_elapsed", 0.0)
        total_time += t
        status = "FAILED" if row.get("_failed") else f"{t:.1f}s"
        print(f"  {row['_name']:<48}  {status:>10}")
    print(f"{'─'*60}")
    print(f"  Total wall time: {total_time:.0f}s  ({total_time / 60:.1f} min)")

    if successful:
        save_csv(all_results, output_path)


if __name__ == "__main__":
    main()
