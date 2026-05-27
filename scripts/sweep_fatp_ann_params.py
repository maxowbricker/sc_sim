#!/usr/bin/env python3
"""
Single-day sweep of FATP-ANN hyperparameters vs Greedy and Static-Composite baselines.

Runs the FULL stratified day (start → finish) via EventSimulator — no RL gym env,
no 8-hour episode window, no greedy warmup.

Grid (9 configs): mu × alpha_scale with use_k_nearest=False (full worker scan).
Plus 2 baselines → 11 total runs.

Usage (from project root):
    conda activate sc
    python scripts/sweep_fatp_ann_params.py

Output:
    outputs/results/fatp_ann_param_sweep_<day>.csv
"""

import copy
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ensure all strategies are registered
import simulator.strategies.greedy  # noqa: E402, F401
import simulator.strategies.composite  # noqa: E402, F401
import simulator.strategies.fatp_ann  # noqa: E402, F401

from config import get_simulation_config  # noqa: E402
from data.loader import load_workers_tasks  # noqa: E402
from simulator.simulation import EventSimulator  # noqa: E402

EVAL_DAY = "496528674@qq.com_20161128"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
DAY_PATH = os.path.join(DATA_ROOT, EVAL_DAY)

STATIC_COMPOSITE_PARAMS = {
    "fairness_weight": 1.0,
    "starvation_weight": 0.2,
    "utility_weight": 1.0,
    "gamma": 0.1,
    "k": 15,
    "soft_threshold": 0.05,
    "enable_diagnostics": False,
    "enable_deferral_tracking": False,
}

MU_VALUES = [0.1, 0.5, 1.0]
ALPHA_SCALE_VALUES = [0.25, 0.5, 1.0]


def extract_metrics(stats):
    wait_times = stats.get("wait_times", [])
    p95_wait = float(np.percentile(wait_times, 95)) if wait_times else 0.0

    return {
        "TAR": stats.get("task_assignment_ratio", 0.0),
        "JFI": stats.get("final_jains_fairness_index", 0.0),
        "Gini": stats.get("final_gini_coefficient", 0.0),
        "Mean Wait (m)": stats.get("avg_wait_time_minutes", 0.0),
        "P95 Wait (m)": p95_wait,
        "Peak Backlog": stats.get("backlog_peak", 0),
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
    }


def run_full_day(workers, tasks, strategy_name, params):
    """Run one strategy over the entire day (all events, no warmup)."""
    sim_config = get_simulation_config()
    sim_config["assignment_strategy"] = strategy_name
    sim_config["strategy_params"] = copy.deepcopy(params)

    sim = EventSimulator(workers, tasks, sim_config=sim_config)
    sim.reset()
    sim.step()  # no duration → run until event queue is empty
    return extract_metrics(sim.get_final_results())


def main():
    if not os.path.exists(DAY_PATH):
        print(f"❌ Data not found: {DAY_PATH}")
        sys.exit(1)

    base_config = get_simulation_config()
    print(f"💿 Loading stratified day data: {EVAL_DAY}")
    workers, tasks = load_workers_tasks(dataset=base_config["dataset"], root_path=DAY_PATH)
    print(f"   {len(workers):,} workers, {len(tasks):,} tasks (full-day simulation, no warmup)\n")

    fatp_grid = list(itertools.product(MU_VALUES, ALPHA_SCALE_VALUES))
    total_runs = 2 + len(fatp_grid)

    print(f"🚀 FATP-ANN param sweep on {EVAL_DAY}")
    print(f"   {len(fatp_grid)} FATP configs + 2 baselines = {total_runs} runs\n")

    all_results = []

    baselines = [
        ("Greedy", "greedy", {}),
        ("Static-Composite", "composite", STATIC_COMPOSITE_PARAMS),
    ]
    for label, strategy_name, params in baselines:
        t0 = time.time()
        try:
            metrics = run_full_day(workers, tasks, strategy_name, params)
            row = {
                "Strategy": label,
                "mu": np.nan,
                "alpha_scale": np.nan,
                "use_k_nearest": np.nan,
                "k": np.nan,
                "Elapsed (s)": round(time.time() - t0, 1),
                **metrics,
            }
            all_results.append(row)
            print(
                f"  ✔️ {label:<18} | Wait: {metrics['Mean Wait (m)']:.2f}m | "
                f"JFI: {metrics['JFI']:.4f} | {row['Elapsed (s)']}s"
            )
        except Exception as e:
            print(f"  ❌ {label:<18} FAILED: {e}")

    for mu, alpha_scale in fatp_grid:
        label = f"FATP-ANN (μ={mu}, α={alpha_scale})"
        params = {
            "mu": mu,
            "alpha_scale": alpha_scale,
            "use_k_nearest": False,
            "k": 15,
        }
        t0 = time.time()
        try:
            metrics = run_full_day(workers, tasks, "fatp_ann", params)
            row = {
                "Strategy": label,
                "mu": mu,
                "alpha_scale": alpha_scale,
                "use_k_nearest": False,
                "k": 15,
                "Elapsed (s)": round(time.time() - t0, 1),
                **metrics,
            }
            all_results.append(row)
            print(
                f"  ✔️ {label:<18} | Wait: {metrics['Mean Wait (m)']:.2f}m | "
                f"JFI: {metrics['JFI']:.4f} | {row['Elapsed (s)']}s"
            )
        except Exception as e:
            print(f"  ❌ {label:<18} FAILED: {e}")

    df = pd.DataFrame(all_results)
    metric_cols = [
        "TAR", "JFI", "Gini", "Mean Wait (m)", "P95 Wait (m)",
        "Peak Backlog", "Avg Pickup (km)",
    ]
    col_order = ["Strategy", "mu", "alpha_scale", "use_k_nearest", "k", "Elapsed (s)"] + metric_cols
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = os.path.join(RESULTS_DIR, f"fatp_ann_param_sweep_{EVAL_DAY.split('_')[-1]}.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"✅ Saved to '{csv_path}'")
    print("=" * 80)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
