#!/usr/bin/env python3
"""
One-shot stats for assignment delay vs passenger wait (not in greedy_baselines_report).

- assignment_delays: seconds from task release → assignment (matching queue), from metrics.
- wait_times: minutes from release → service start (pickup), for completed tasks.

Usage:
    python scripts/sample_assignment_wait_stats.py
    python scripts/sample_assignment_wait_stats.py --stratified false --day 496528674@qq.com_20161128
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as config_module
from config import get_simulation_config
from data.loader import load_workers_tasks
from simulator.simulation import EventSimulator


def pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia"),
    )
    parser.add_argument("--day", default="496528674@qq.com_20161128")
    parser.add_argument(
        "--stratified",
        type=lambda x: x.lower() == "true",
        default=None,
        help="true/false override for DATA_SAMPLING.use_stratified_sampling",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root.lstrip("./"))

    if args.stratified is not None:
        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified

    day_path = os.path.join(data_root, args.day)
    workers, tasks = load_workers_tasks("didi", root_path=day_path)
    if not tasks:
        print("No tasks")
        return 1

    cfg = get_simulation_config()
    cfg["assignment_strategy"] = "greedy"
    cfg["strategy_params"] = {"enable_deferral_tracking": False}

    sim = EventSimulator(workers, tasks, cfg)
    sim.reset()
    sim.step()
    r = sim.get_final_results()

    ad = np.asarray(r.get("assignment_delays") or [], dtype=float)
    wt = np.asarray(r.get("wait_times") or [], dtype=float)

    print(f"day={args.day}  workers={len(workers)}  tasks={len(tasks)}")
    print()
    print("assignment_delay = time from task RELEASE to ASSIGNMENT (seconds).")
    print("  (NOT worker idle time between jobs.)")
    if len(ad):
        print(
            f"  assignment_delays sec:  n={len(ad)}  mean={ad.mean():.1f}  "
            f"p50={pct(ad, 50):.1f}  p90={pct(ad, 90):.1f}  max={ad.max():.1f}"
        )
    else:
        print("  assignment_delays: (empty — check on_task_assigned / task.start_time)")
    print()
    print("wait_time = time from RELEASE to SERVICE START / pickup (minutes).")
    if len(wt):
        print(
            f"  wait_times min:  n={len(wt)}  mean={wt.mean():.3f}  "
            f"p50={pct(wt, 50):.3f}  p90={pct(wt, 90):.3f}  max={wt.max():.3f}"
        )
    print()
    print("Compare to greedy_baselines_report avg_wait_m (same wait metric, end-of-run mean).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
