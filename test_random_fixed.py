#!/usr/bin/env python3
"""Quick test to verify Random strategy works with correct calling convention."""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
import copy

print("=" * 80)
print("TESTING FIXED RANDOM STRATEGY")
print("=" * 80)

# Load 3-hour peak dataset
print("\nLoading data...")
data_path = project_root / "data" / "didi"
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")

# Sample small dataset
workers = copy.deepcopy(all_workers[:500])
tasks = copy.deepcopy(all_tasks[:2000])
print(f"✅ Sampled {len(workers)} workers, {len(tasks)} tasks")

# Configure Random strategy
random_config = {
    'assignment_strategy': 'random_assign',
    'strategy_params': {'k': 15}
}

print("\n🎲 Running Random strategy...")
try:
    random_summary = run_simulation(workers, tasks, sim_config=random_config)
    print(f"✅ SUCCESS!")
    print(f"   TAR: {random_summary['task_assignment_ratio']:.1%}")
    print(f"   JFI: {random_summary['jains_fairness_index']:.3f}")
    print(f"   Wait: {random_summary['mean_wait_time_minutes']:.2f} min")
    print(f"   Completed: {random_summary['completed_tasks']:,}/{len(tasks):,} tasks")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

