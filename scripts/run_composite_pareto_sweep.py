#!/usr/bin/env python3
"""
Composite Strategy — Pareto Frontier Sweep (Didi 20161109)

Sweeps fairness_weight × starvation_weight with all other params fixed,
recording (JFI, TAR, avg_wait) for every config to produce a JFI vs wait
Pareto frontier plot.

Grid:
  fairness_weight  : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  (11 values)
  starvation_weight: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]        (10 values)
  Total: 110 configs  —  estimated ~2.5 hours at ~80s/run

Fixed params (from soft_threshold sensitivity test):
  utility_weight=1.0, gamma=0.1, k=15, soft_threshold=0.0

Also runs a Greedy reference row at startup for context.

Usage:
    caffeinate python scripts/run_composite_pareto_sweep.py
    caffeinate python scripts/run_composite_pareto_sweep.py --resume       # continue after ctrl-C
    caffeinate python scripts/run_composite_pareto_sweep.py --output my.csv
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

DATA_ROOT  = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY = "496528674@qq.com_20161109"

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

FW_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # 11
SW_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 11 (0.0 = no starvation signal)

# Fixed params (soft_threshold=0.0 chosen from sensitivity test — best JFI, no masking)
FIXED = dict(
    utility_weight=1.0,
    gamma=0.1,
    k=15,
    soft_threshold=0.0,
)

TIMEOUT_SEC = 300  # 5 min hard cap — normal runs take ~80–110s

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "fairness_weight", "starvation_weight",
    "utility_weight", "gamma", "k", "soft_threshold",
    "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
    "Completed", "Total", "elapsed_s",
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _sim_thread(sim, exc_holder):
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_config(
    workers_template,
    tasks_template,
    strategy_key: str,
    params: dict,
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

        thread = threading.Thread(
            target=_sim_thread, args=(sim, exc_holder), daemon=True
        )
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
# Resume helpers
# ---------------------------------------------------------------------------

def _config_key(fw: float, sw: float) -> str:
    return f"{fw:.2f}_{sw:.2f}"


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                done.add(_config_key(float(row["fairness_weight"]),
                                     float(row["starvation_weight"])))
            except (KeyError, ValueError):
                pass
    return done


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

COL_W  = 28
STAT_W = 13

DISPLAY_KEYS = [
    "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
]


def _fmt(v: Any, key: str) -> str:
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate"):
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def _header() -> str:
    return (f"{'Config':<{COL_W}}"
            + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS)
            + f"  {'Time (s)':>8}")


def _sep() -> str:
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 10)


def _row(name: str, m: Dict[str, Any]) -> str:
    cells = "".join(f" {_fmt(m.get(k, 0), k):>{STAT_W}}" for k in DISPLAY_KEYS)
    return f"{name:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>8.1f}"


# ---------------------------------------------------------------------------
# Final Pareto summary
# ---------------------------------------------------------------------------

def print_pareto_summary(results: List[Dict]) -> None:
    """Print the Pareto-optimal configs (maximise JFI subject to wait constraint)."""
    if not results:
        return

    # Sort by JFI descending, then wait ascending
    ranked = sorted(results, key=lambda r: (-r["JFI (tasks)"], r["Avg Wait (m)"]))

    print(f"\n{'=' * 80}")
    print("  TOP 10 CONFIGS by JFI (tasks)")
    print(f"{'=' * 80}")
    print(_sep())
    print(_header())
    print(_sep())
    for r in ranked[:10]:
        label = f"fw={r['fairness_weight']:.1f}  sw={r['starvation_weight']:.1f}"
        print(_row(label, r))
    print(_sep())

    # Pareto frontier: for each JFI level, the config with lowest wait
    print(f"\n{'=' * 80}")
    print("  PARETO FRONTIER  (non-dominated: highest JFI at each wait level)")
    print(f"{'=' * 80}")
    print(_sep())
    print(_header())
    print(_sep())
    pareto: List[Dict] = []
    best_jfi_seen = -1.0
    for r in sorted(results, key=lambda x: x["Avg Wait (m)"]):
        if r["JFI (tasks)"] > best_jfi_seen:
            pareto.append(r)
            best_jfi_seen = r["JFI (tasks)"]
    for r in pareto:
        label = f"fw={r['fairness_weight']:.1f}  sw={r['starvation_weight']:.1f}"
        print(_row(label, r))
    print(_sep())

    # Highlight the selected operating point
    selected = next(
        (r for r in results
         if abs(r["fairness_weight"] - 1.0) < 1e-6 and abs(r["starvation_weight"] - 0.2) < 1e-6),
        None,
    )
    if selected:
        print(f"\n  Paper config (fw=1.0, sw=0.2):  "
              f"JFI={selected['JFI (tasks)']:.4f}  "
              f"wait={selected['Avg Wait (m)']:.3f}m  "
              f"TAR={selected['TAR']:.4f}")
        on_frontier = any(
            abs(r["fairness_weight"] - 1.0) < 1e-6
            and abs(r["starvation_weight"] - 0.2) < 1e-6
            for r in pareto
        )
        print(f"  On Pareto frontier: {'YES ✓' if on_frontier else 'NO — check frontier configs above'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already written to --output CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path (default: pareto_sweep_20161109.csv in project root)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC,
                        help=f"Per-run timeout in seconds (default: {TIMEOUT_SEC})")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        PROJECT_ROOT, "pareto_sweep_20161109.csv"
    )

    # Full grid
    all_configs = [(fw, sw) for fw in FW_VALUES for sw in SW_VALUES]

    done = load_done(output_path) if args.resume else set()
    pending = [(fw, sw) for (fw, sw) in all_configs
               if _config_key(fw, sw) not in done]

    n_total   = len(all_configs)
    n_pending = len(pending)
    est_min   = n_pending * 85 / 60

    print("=" * 80)
    print("  Composite Pareto Sweep — Didi 20161109")
    print(f"  Grid:       fairness_weight × starvation_weight  ({len(FW_VALUES)} × {len(SW_VALUES)} = {n_total})")
    print(f"  Fixed:      utility_weight=1.0  gamma=0.1  k=15  soft_threshold=0.0")
    print(f"  Total:      {n_total} configs   Done: {len(done)}   Pending: {n_pending}")
    print(f"  Est. time:  ~{est_min:.0f} min  (~{est_min/60:.1f} h)")
    print(f"  Output:     {output_path}")
    if args.resume and done:
        print(f"  Resume:     {len(done)} config(s) already cached — skipping.")
    print("=" * 80)

    # Load data once
    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_template):,} workers | {len(tasks_template):,} tasks\n")

    # Run Greedy reference first (always, not cached)
    print("  Running Greedy reference ...")
    t_ref = time.time()
    greedy_m = run_config(workers_template, tasks_template, "greedy", {}, args.timeout)
    if greedy_m:
        print(
            f"  Greedy:  TAR={greedy_m['TAR']:.4f}  "
            f"JFI={greedy_m['JFI (tasks)']:.4f}  "
            f"wait={greedy_m['Avg Wait (m)']:.2f}m  "
            f"[{greedy_m['elapsed_s']:.1f}s]"
        )
    else:
        print("  Greedy: FAILED — continuing without reference")
    print()

    # Open CSV (append if resuming, write fresh otherwise)
    file_exists = os.path.exists(output_path) and args.resume
    sweep_results: List[Dict] = []

    with open(output_path, "a" if file_exists else "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        t_sweep_start = time.time()

        for idx, (fw, sw) in enumerate(pending, 1):
            elapsed_total = time.time() - t_sweep_start
            remaining     = n_pending - idx + 1
            eta_min = (elapsed_total / max(idx - 1, 1)) * remaining / 60 if idx > 1 else est_min

            label = f"fw={fw:.1f}  sw={sw:.1f}"
            print(
                f"  [{idx:>3}/{n_pending}]  {label:<20}  "
                f"(ETA ~{eta_min:.0f}m remaining)",
                end="  ",
                flush=True,
            )

            m = run_config(
                workers_template, tasks_template,
                "composite",
                {**FIXED, "fairness_weight": fw, "starvation_weight": sw},
                args.timeout,
            )

            if m is None:
                print("TIMEOUT / FAILED — skipping")
                continue

            print(
                f"TAR={m['TAR']:.4f}  "
                f"JFI={m['JFI (tasks)']:.4f}  "
                f"wait={m['Avg Wait (m)']:.2f}m  "
                f"[{m['elapsed_s']:.1f}s]"
            )

            row = {
                "fairness_weight":   fw,
                "starvation_weight": sw,
                **FIXED,
                **m,
            }
            writer.writerow(row)
            f.flush()

            m["fairness_weight"]   = fw
            m["starvation_weight"] = sw
            sweep_results.append(m)

    # Load any previously cached results for the full summary
    if args.resume and done:
        with open(output_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    cached = {k: float(row[k]) for k in FIELDNAMES
                              if k not in ("elapsed_s",) and row.get(k) not in (None, "")}
                    cached["elapsed_s"] = float(row.get("elapsed_s", 0))
                    sweep_results.append(cached)
                except (KeyError, ValueError):
                    pass

    total_elapsed = time.time() - t_sweep_start
    print(f"\n  Sweep complete.  Total time: {total_elapsed:.0f}s  ({total_elapsed/60:.1f} min)")
    print(f"  Results saved → {output_path}")

    if sweep_results:
        print_pareto_summary(sweep_results)


if __name__ == "__main__":
    main()
