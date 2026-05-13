#!/usr/bin/env python3
"""
Quick smoke test for mmd_batch (CR-11–style batch matching baseline) on the standard eval day.
Disables stratified sampling so the full raw day is simulated.
"""
import os
import sys

# Repo root = parent of scripts/ — works no matter where you run `python` from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import config as config_module
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation


def main():
    # 1. Turn off stratified sampling so we get the full day's real chaos
    config_module.DATA_SAMPLING["use_stratified_sampling"] = False

    # 2. Configure MMD Batch (batch_interval_minutes is reserved for future use; matching runs each event)
    sim_config = {
        "assignment_strategy": "mmd_batch",
        "strategy_params": {
            "batch_interval_minutes": 5,
        },
        "dataset": "didi",
    }

    # 3. Load the standard evaluation day (path anchored to repo root, not cwd)
    day = "496528674@qq.com_20161128"
    day_path = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia", day)

    print("=" * 60)
    print("🧪 Testing MMD Batch Strategy (CR-11–style baseline)")
    print(f"Day: {day}")
    print("=" * 60)

    workers, tasks = load_workers_tasks("didi", root_path=day_path)
    print(f"Loaded {len(workers)} workers and {len(tasks)} tasks.")

    # 4. Run Simulation
    print("\nRunning simulation (Hungarian per assignment wave; full day may take a while)...")
    result = run_simulation(workers=workers, tasks=tasks, sim_config=sim_config)

    # 5. Output Baseline Metrics
    print("\n" + "=" * 60)
    print("✅ MMD-BB SIMULATION COMPLETE")
    print("-" * 60)
    print(f"Completed Tasks:  {result.get('completed_tasks', 0)}")
    print(f"JFI (Fairness):   {result.get('final_jains_fairness_index', 0):.4f}")
    print(f"Gini coefficient: {result.get('final_gini_coefficient', 0.0):.4f}")
    print(f"Avg Wait Time:    {result.get('avg_wait_time_minutes', 0):.2f} mins")
    print(f"Avg pickup dist:  {result.get('avg_pickup_distance_km', 0.0):.2f} km")
    print(f"Peak Backlog:     {result.get('backlog_peak', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
