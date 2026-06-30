#!/usr/bin/env python3
"""
Section 5.3 — Computational Efficiency & Scalability
Experiment B: Vary Task Volume |T|, Fix Fleet Size |W|

Holds the worker fleet at a fixed 10,000 workers (stratified sample) and
increases the task volume from 50,000 to the full 224,219 using STRATIFIED
temporal sampling.

When |W| is fixed, both O(W) and O(k log W) strategies scale approximately
linearly in |T| (more tasks → more events). The important distinction is the
SLOPE: O(W) strategies (Greedy, LAF) have a steeper per-event cost than
O(k log W) strategies (k-NLF, Composite), so their runtime lines diverge as
|T| grows. This provides complementary evidence to Experiment A's fleet-
scaling test.

Worker count: 10,000  (stratified sample, same seed across all runs)
Task counts:  [50,000 · 100,000 · 150,000 · 200,000 · 224,219]
Bins:         288  (5-minute temporal bins; preserves intra-day density)
Seed:         42

Output:
    results/s4_robustness/scalability_tasks.csv

Usage:
    python scripts/experiments/s4_robustness/run_scalability_tasks.py
    python scripts/experiments/s4_robustness/run_scalability_tasks.py --timeout 1200
    python scripts/experiments/s4_robustness/run_scalability_tasks.py --seed 99
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

DATA_ROOT   = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY  = "496528674@qq.com_20161109"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "s53_scalability")

# Fixed worker fleet size across all task-volume runs
WORKER_COUNT = 10_000

# Task volumes to sweep; fleet stays at WORKER_COUNT
TASK_COUNTS = [50_000, 100_000, 150_000, 200_000, 224_219]

# Temporal stratification bins (288 × 5 min = 24 h)
NUM_BINS = 288

TIMEOUT_SEC = 900  # 15 min hard cap per run

# Same strategy set as Experiment A for consistent comparison
STRATEGIES: List[tuple] = [
    ("k-NLF (k=15)",       "knlf",      {"k": 15}),
    ("Composite (static)", "composite", {
        "fairness_weight": 1.6, "starvation_weight": 0.0,
        "utility_weight": 1.0, "gamma": 0.1, "k": 15, "soft_threshold": 0.0,
    }),
    ("Greedy",             "greedy",    {}),
    ("LAF",                "laf",       {}),
    ("FATP-ANN",           "fatp_ann",  {
        "mu": 1.5, "alpha_scale": 0.5, "use_k_nearest": True, "k": 15,
    }),
]

FIELDNAMES = [
    "n_workers", "n_tasks", "strategy",
    "elapsed_s", "TAR", "JFI (tasks)", "Avg Wait (m)",
]


# ---------------------------------------------------------------------------
# Simulation runner (identical to fleet script)
# ---------------------------------------------------------------------------

def _sim_thread(sim, exc_holder: dict) -> None:
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_one(
    workers_subset: list,
    tasks_list: list,
    strategy_key: str,
    strategy_params: dict,
    timeout_sec: float = TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    cfg = create_composite_config(
        assignment_strategy=strategy_key, **strategy_params
    )
    t0 = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator
        sim = EventSimulator(
            copy.deepcopy(workers_subset),
            copy.deepcopy(tasks_list),
            cfg,
        )
        sim.reset()
        thread = threading.Thread(target=_sim_thread, args=(sim, exc_holder), daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            return None  # timed out
        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        print(f"    FAILED: {exc}")
        return None

    completed  = stats.get("completed_tasks", 0)
    total      = stats.get("total_tasks", 1)
    wait_times = stats.get("wait_times", [])

    return {
        "n_workers":    len(workers_subset),
        "n_tasks":      total,
        "strategy":     strategy_key,
        "elapsed_s":    time.time() - t0,
        "TAR":          completed / total if total else 0.0,
        "JFI (tasks)":  stats.get("final_jains_fairness_index", 0.0),
        "Avg Wait (m)": float(np.mean(wait_times)) if wait_times else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC,
                        help="Per-run timeout in seconds (default: 900)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Stratified sampling seed (default: 42)")
    parser.add_argument("--output",  type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "scalability_tasks.csv")

    n_points     = len(TASK_COUNTS)
    n_strategies = len(STRATEGIES)
    n_total      = n_points * n_strategies

    print("=" * 72)
    print("  §5.3 Scalability — Experiment B: Vary Task Volume |T|, Fix |W|")
    print(f"  Dataset:    {TARGET_DAY}")
    print(f"  Workers:    fixed at {WORKER_COUNT:,} (stratified)")
    print(f"  Tasks:      {TASK_COUNTS}")
    print(f"  Strategies: {', '.join(s[0] for s in STRATEGIES)}")
    print(f"  Total runs: {n_total}  |  Timeout: {args.timeout}s  |  Seed: {args.seed}")
    print(f"  Output:     {output_path}")
    print("=" * 72)

    # ── Load full dataset once ───────────────────────────────────────────────
    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    all_workers, all_tasks = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(all_workers):,} workers | {len(all_tasks):,} tasks  (full load)")

    # ── Stratified sampling — one worker sample reused across all task volumes
    print(f"\n  Stratified sampling ({NUM_BINS} bins, seed={args.seed}) ...")
    print(f"  Building task samples for volumes: {TASK_COUNTS} ...")

    # Sample tasks at all target volumes; workers sampled once (WORKER_COUNT)
    # We call the sampler once per target_tasks value so each task sample is
    # independently stratified to the correct volume.
    task_samples: Dict[int, list] = {}
    workers_subset: Optional[list] = None  # reuse the same worker sample

    for target_t in TASK_COUNTS:
        sampled_tasks, worker_samples = stratified_temporal_sample(
            all_workers=all_workers,
            all_tasks=all_tasks,
            target_tasks=target_t,
            worker_counts=[WORKER_COUNT],
            num_bins=NUM_BINS,
            seed=args.seed,  # same seed → same worker sample every call
        )
        task_samples[target_t] = sampled_tasks
        if workers_subset is None:
            workers_subset = worker_samples[WORKER_COUNT]
        print(f"    tasks[{target_t:>7,}] → {len(sampled_tasks):,} sampled  |  "
              f"workers → {len(workers_subset):,}")

    # ── Run sweep ────────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    run_idx = 0

    COL_W = 26
    hdr = (f"  {'Task vol. / Strategy':<{COL_W}}"
           f"  {'TAR':>7}  {'JFI':>7}  {'Wait(m)':>8}  {'Time(s)':>8}")
    sep = "  " + "─" * (len(hdr) - 2)

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for target_t in TASK_COUNTS:
            tasks_list = task_samples[target_t]
            print(f"\n{sep}")
            print(f"  Task volume: {len(tasks_list):,}  |  Fleet: {len(workers_subset):,} workers")
            print(sep)
            print(hdr)
            print(sep)

            for strat_name, strat_key, strat_params in STRATEGIES:
                run_idx += 1

                m = run_one(
                    workers_subset, tasks_list,
                    strat_key, strat_params,
                    timeout_sec=args.timeout,
                )

                if m is None:
                    print(f"  [{run_idx:>2}/{n_total}] {strat_name:<{COL_W}}  TIMEOUT / FAILED")
                    row = {"n_workers": len(workers_subset), "n_tasks": len(tasks_list),
                           "strategy": strat_name,
                           "elapsed_s": args.timeout, "TAR": None,
                           "JFI (tasks)": None, "Avg Wait (m)": None}
                else:
                    print(
                        f"  [{run_idx:>2}/{n_total}] {strat_name:<{COL_W}}"
                        f"  {m['TAR']:>7.4f}"
                        f"  {m['JFI (tasks)']:>7.4f}"
                        f"  {m['Avg Wait (m)']:>8.2f}"
                        f"  {m['elapsed_s']:>8.1f}"
                    )
                    row = {**m, "strategy": strat_name}
                    all_results.append(row)

                writer.writerow(row)
                f.flush()

    print(f"\n{'=' * 72}")
    print(f"  Sweep complete.  {len(all_results)}/{n_total} runs succeeded.")
    print(f"  Results saved → {output_path}")
    print("=" * 72)

    # ── Quick summary table ──────────────────────────────────────────────────
    if all_results:
        print(f"\n  Runtime summary (seconds):")
        print(f"  {'Strategy':<26}" + "".join(f"  {tc:>8,}" for tc in TASK_COUNTS))
        print("  " + "─" * (26 + 10 * len(TASK_COUNTS)))
        for strat_name, _, _ in STRATEGIES:
            row_str = f"  {strat_name:<26}"
            for tc in TASK_COUNTS:
                match = [r for r in all_results
                         if r["strategy"] == strat_name and r["n_tasks"] == tc]
                row_str += f"  {match[0]['elapsed_s']:>8.1f}" if match else f"  {'---':>8}"
            print(row_str)


if __name__ == "__main__":
    main()
