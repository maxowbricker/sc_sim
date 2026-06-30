#!/usr/bin/env python3
"""
Greedy k-sweep: test k=1, 3, 10, 50 on DiDi 20161109.

Measures TAR, JFI, wait time, avg pickup distance, and wall-clock runtime
to determine the lowest k that preserves full greedy correctness.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks
from simulator.simulation import EventSimulator

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY = "20161109"

K_VALUES = [1, 3, 10, 50]


def run_greedy_k(workers_template, tasks_template, k: int):
    cfg = create_composite_config(assignment_strategy="greedy", k=k)
    sim = EventSimulator(workers_template, tasks_template, cfg)
    sim.reset()
    t0 = time.time()
    sim.step()
    elapsed = time.time() - t0
    stats = sim.get_final_results()
    workers = list(sim.state.all_workers_map.values())

    completed = stats.get("completed_tasks", 0)
    total     = stats.get("total_tasks", 1)
    tar       = completed / total
    jfi       = stats.get("final_jains_fairness_index", 0.0)
    wait_times = stats.get("wait_times", [])
    avg_wait  = float(np.mean(wait_times)) if wait_times else 0.0
    avg_pick  = stats.get("avg_pickup_distance_km", 0.0)
    backlog   = stats.get("backlog_peak", 0)

    return {
        "k":         k,
        "completed": completed,
        "total":     total,
        "TAR":       tar,
        "JFI":       jfi,
        "avg_wait":  avg_wait,
        "avg_pick":  avg_pick,
        "backlog":   backlog,
        "elapsed":   elapsed,
    }


def main():
    day_dirs = sorted(d for d in os.listdir(DATA_ROOT)
                      if os.path.isdir(os.path.join(DATA_ROOT, d)) and TARGET_DAY in d)
    if not day_dirs:
        print(f"ERROR: no folder matching {TARGET_DAY} in {DATA_ROOT}")
        sys.exit(1)

    day_path = os.path.join(DATA_ROOT, day_dirs[0])
    print(f"Loading {day_dirs[0]} ...")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_template):,} workers | {len(tasks_template):,} tasks\n")

    results = []
    for k in K_VALUES:
        print(f"  Running greedy k={k} ...", end=" ", flush=True)
        r = run_greedy_k(workers_template, tasks_template, k)
        results.append(r)
        print(f"TAR={r['TAR']:.4f}  JFI={r['JFI']:.4f}  "
              f"wait={r['avg_wait']:.2f}m  pick={r['avg_pick']:.3f}km  "
              f"backlog={r['backlog']}  [{r['elapsed']:.1f}s]")

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 8
    print(f"\n{'─'*72}")
    print(f"  Greedy k-sweep — DiDi {TARGET_DAY}")
    print(f"{'─'*72}")
    hdr = f"  {'k':>4}  {'TAR':>7}  {'JFI':>7}  {'Wait(m)':>8}  {'Pick(km)':>9}  {'Backlog':>8}  {'Time(s)':>8}"
    print(hdr)
    print(f"{'─'*72}")
    ref = results[0]  # k=1 as reference
    for r in results:
        tar_diff = r["TAR"] - ref["TAR"]
        jfi_diff = r["JFI"] - ref["JFI"]
        tag = "  <-- reference" if r["k"] == ref["k"] else \
              f"  ΔTAR={tar_diff:+.4f}  ΔJFI={jfi_diff:+.4f}"
        print(f"  {r['k']:>4}  {r['TAR']:.4f}  {r['JFI']:.4f}  "
              f"{r['avg_wait']:>8.2f}  {r['avg_pick']:>9.3f}  "
              f"{r['backlog']:>8}  {r['elapsed']:>8.1f}{tag}")
    print(f"{'─'*72}\n")


if __name__ == "__main__":
    main()
