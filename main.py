"""Convenience entry-point.

Run the simulator using the settings defined in ``config.py``.

Usage (from project root):
    $ python main.py

Optional CLI flags:
    --dataset didi|synthetic  (overrides config dataset)
    --root    PATH            (overrides config root_path)
    --strategy STRATEGY       (overrides config assignment_strategy)
"""

import argparse
import time
from pathlib import Path

from config import create_composite_config, get_data_sampling_config
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial-crowdsourcing simulator runner")
    p.add_argument("--dataset", help="Override dataset key in config.py")
    p.add_argument("--root", help="Override root_path in config.py")
    p.add_argument("--strategy", help="Override assignment_strategy in config.py")
    return p.parse_args()


def main():
    args = parse_args()

    # Build configuration with command line overrides
    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.root:
        overrides["root_path"] = args.root
    if args.strategy:
        overrides["assignment_strategy"] = args.strategy
    
    # 1. UNIFIED CONFIGURATION
    # Our optimized create_composite_config handles ALL strategies safely!
    cfg = create_composite_config(**overrides)
    sampling_cfg = get_data_sampling_config()

    print("=" * 60)
    print(f"STARTING SIMULATION: {cfg['assignment_strategy'].upper()}")
    print("=" * 60)

    # 2. LOAD DATA
    start_time = time.time()
    workers, tasks = load_workers_tasks(cfg["dataset"], cfg.get("root_path"))
    
    # (Note: If you toggle use_stratified_sampling=True in config.py, 
    # you can wire your stratified_sampler.py here!)
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s ({len(workers):,} workers, {len(tasks):,} tasks)")

    # 3. RUN SIMULATION
    print("\nRunning physics engine...")
    sim_start_time = time.time()
    
    results = run_simulation(
        workers,
        tasks,
        sim_config=cfg,
        # Pass the extracted strategy parameters directly to the engine
        **cfg["strategy_params"]
    )
    
    sim_time = time.time() - sim_start_time

    # 4. RESULTS DASHBOARD
    wait_times = results.get("wait_times", [0])
    avg_wait = sum(wait_times) / max(1, len(wait_times))

    print("\n" + "=" * 60)
    print(f"SIMULATION COMPLETE IN {sim_time:.2f}s")
    print("=" * 60)
    print(f"Tasks Completed:       {results.get('completed_tasks', 0):,}")
    print(f"Jain's Fairness Index: {results.get('final_jains_fairness_index', 0.0):.4f}")
    print(f"Utility Difference:    {results.get('final_utility_difference_tasks', 0.0):.2f}")
    print(f"Avg Wait Time (min):   {avg_wait:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()