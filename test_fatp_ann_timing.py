"""
Full-scale timing test for FATP-ANN strategy.
Tests performance with 4K workers / 20K tasks.
"""

import sys
from pathlib import Path
import time
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
import pandas as pd
import numpy as np

print("=" * 80)
print("FATP-ANN FULL-SCALE TIMING TEST")
print("=" * 80)
print()

# Load data
data_path = project_root / "data" / "didi"
print(f"📊 Loading data from: {data_path}")
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
print(f"✅ Loaded {len(all_workers)} workers and {len(all_tasks)} tasks")
print()

# Stratified sampling to 4K workers / 20K tasks
print("🎯 Sampling 4000 workers and 20000 tasks using stratified temporal sampling...")

# Sample workers
worker_times = [w.release_time for w in all_workers]
workers_df = pd.DataFrame({'worker': all_workers, 'time': worker_times})
workers_df['time_bin'] = pd.cut(workers_df['time'], bins=10)
sampled_workers_df = workers_df.groupby('time_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 400), random_state=42)
)
test_workers = sampled_workers_df['worker'].tolist()[:4000]

# Sample tasks
task_times = [t.release_time for t in all_tasks]
tasks_df = pd.DataFrame({'task': all_tasks, 'time': task_times})
tasks_df['time_bin'] = pd.cut(tasks_df['time'], bins=10)
sampled_tasks_df = tasks_df.groupby('time_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 2000), random_state=42)
)
test_tasks = sampled_tasks_df['task'].tolist()[:20000]

print(f"✅ Sampled {len(test_workers)} workers and {len(test_tasks)} tasks")
print()

# Configure FATP-ANN with k-NN optimization
config = {
    'assignment_strategy': 'fatp_ann',
    'mu': 0.1,
    'alpha_scale': 1.0,
    'use_k_nearest': True,  # k-NN optimization enabled
    'k': 15
}

print("📋 Configuration:")
print(f"   Strategy: {config['assignment_strategy']}")
print(f"   mu (decay): {config['mu']}")
print(f"   alpha_scale: {config['alpha_scale']}")
print(f"   use_k_nearest: {config['use_k_nearest']}")
print()

# Run simulation with timing
print("🚀 Starting FATP-ANN simulation...")
print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

start_time = time.time()

try:
    summary = run_simulation(
        workers=test_workers,
        tasks=test_tasks,
        sim_config=config
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print()
    print("=" * 80)
    print("✅ SIMULATION COMPLETE")
    print("=" * 80)
    print()
    print(f"⏱️  Total Runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print()
    print("📊 Results:")
    print(f"   Completed tasks: {summary['completed_tasks']:,} / {len(test_tasks):,}")
    print(f"   Task Assignment Ratio: {summary.get('task_assignment_ratio', 0):.2f}%")
    print(f"   Jain's Fairness Index: {summary.get('final_jains_fairness_index', 0):.4f}")
    print(f"   Avg wait time: {summary.get('avg_wait_time_minutes', 0):.2f} min")
    print(f"   Avg travel distance: {summary.get('avg_travel_distance_km', 0):.2f} km")
    print()
    
    # Performance metrics
    events_per_second = (summary['completed_tasks'] * 3) / elapsed  # Rough estimate
    print("⚡ Performance:")
    print(f"   Seconds per task: {elapsed / summary['completed_tasks']:.3f}s")
    print(f"   Tasks per minute: {summary['completed_tasks'] / (elapsed/60):.1f}")
    print()
    
except Exception as e:
    print()
    print("=" * 80)
    print("❌ SIMULATION FAILED")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("TIMING TEST COMPLETE")
print("=" * 80)
print()
print("💡 Next Steps:")
print("   - If runtime is acceptable (< 10 min), proceed with tuning experiment")
print("   - If too slow, enable use_k_nearest=True optimization")
print("   - Typical tuning experiment: 15-20 configs × 7-10 min = 2-3 hours")
print()

