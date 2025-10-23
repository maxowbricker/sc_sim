#!/usr/bin/env python3
"""
Rerun ONLY the Random baseline for Experiment 015 after fixing the calling convention bug.
"""

import sys
import os
import json
import pandas as pd
import copy
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

# Configuration
EXPERIMENT_ID = "015"
DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
EXISTING_RUN_DIR = os.path.join(DATA_OUTPUT_DIR, "exp_015_20251024_002521")

# Robust parameters
ROBUST_PARAMS = {
    'soft_threshold': 0.0,
    'starvation_weight': 0.5,
    'normalize_scores': True,
    'k': 15,
}

print("=" * 80)
print("EXPERIMENT 015: RERUN RANDOM BASELINE ONLY")
print("=" * 80)
print(f"Output Directory: {EXISTING_RUN_DIR}")
print("=" * 80)

# ============================================================================
# LOAD AND SAMPLE DATA (same as original experiment)
# ============================================================================
print("\nLOADING DATA...")
print("=" * 80)

print(f"\n[STEP 1] Loading 3-hour peak dataset...")
data_path = project_root / "data" / "didi"
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
print(f"  ✅ Loaded {len(all_workers):,} workers and {len(all_tasks):,} tasks")

# Get time ranges
worker_start = min(w.release_time for w in all_workers)
worker_end = max(w.deadline for w in all_workers)
task_start = min(t.release_time for t in all_tasks)
task_end = max(t.release_time for t in all_tasks)

print(f"\n[STEP 2] Time Ranges:")
print(f"  Task window: {task_start} to {task_end}")
print(f"  Worker window: {worker_start} to {worker_end}")

overlap_start = max(task_start, worker_start)
overlap_end = min(task_end, worker_end)
overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600

print(f"\n  ✅ Overlap window: {overlap_start} to {overlap_end}")
print(f"     Duration: {overlap_hours:.2f} hours")

# Stratified temporal sampling
print(f"\n[STEP 3] Stratified Temporal Sampling...")

TARGET_WORKERS = 4000
TARGET_TASKS = 20000
NUM_BINS = 12

# Sample workers
print(f"\n[STEP 3.1] Sampling {TARGET_WORKERS:,} workers...")
bin_duration = (overlap_end - overlap_start) / NUM_BINS
workers_per_bin = TARGET_WORKERS // NUM_BINS

sampled_workers = []
for i in range(NUM_BINS):
    bin_start = overlap_start + i * bin_duration
    bin_end = bin_start + bin_duration
    
    bin_workers = [w for w in all_workers if bin_start <= w.release_time < bin_end]
    
    if len(bin_workers) > workers_per_bin:
        bin_sample = pd.Series(bin_workers).sample(n=workers_per_bin, random_state=42).tolist()
    else:
        bin_sample = bin_workers
    
    sampled_workers.extend(bin_sample)

print(f"  ✅ Sampled {len(sampled_workers):,} workers")

# Sample tasks
print(f"\n[STEP 3.2] Sampling {TARGET_TASKS:,} tasks...")
task_sample_rate = TARGET_TASKS / len(all_tasks)
sampled_tasks = pd.Series(all_tasks).sample(n=min(TARGET_TASKS, len(all_tasks)), random_state=42).tolist()
print(f"  ✅ Sampled {len(sampled_tasks):,} tasks")

print("\n" + "=" * 80)
print(f"✅ Loaded {len(sampled_workers):,} workers and {len(sampled_tasks):,} tasks")
print("=" * 80)

# ============================================================================
# RUN RANDOM BASELINE
# ============================================================================

print("\n🎲 Experiment 003/033 - Random_Baseline")
print("   Strategy: random_assign")
print("   Params: {'k': 15}")

# Deep copy data for this run
exp_workers = copy.deepcopy(sampled_workers)
exp_tasks = copy.deepcopy(sampled_tasks)

# Configure simulation
sim_config = {
    'assignment_strategy': 'random_assign',
    'strategy_params': {'k': 15}
}

# Run simulation
try:
    summary = run_simulation(exp_workers, exp_tasks, sim_config=sim_config)
    
    # Add experiment metadata
    summary['experiment_id'] = 'exp_015_003'
    summary['experiment_name'] = 'Random_Baseline'
    summary['strategy'] = 'random_assign'
    summary['parameters'] = {'k': 15}
    
    # Print results
    print(f"   ✅ Completed: {summary['completed_tasks']:,}/{len(sampled_tasks):,} tasks")
    print(f"   JFI: {summary['jains_fairness_index']:.3f}")
    print(f"   Gini: {summary['gini_coefficient']:.3f}")
    print(f"   Wait: {summary['mean_wait_time_minutes']:.2f} min")
    
    # Check if EWMA temporal history exists
    if 'ewma_temporal_history' in summary:
        print(f"   EWMA snapshots: {len(summary['ewma_temporal_history'])}")
        print(f"   EWMA Final: {summary['ewma_final_mean']:.4f}")
    
    # Save individual summary
    summary_path = os.path.join(EXISTING_RUN_DIR, "exp_003_Random_Baseline_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   💾 Saved: exp_003_Random_Baseline_summary.json")
    
    # Update aggregate CSV
    print("\n📊 Updating aggregate CSV...")
    csv_path = os.path.join(DATA_OUTPUT_DIR, "experiment_015_aggregate_results.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Remove old Random_Baseline entry if it exists
        df = df[df['exp_name'] != 'Random_Baseline']
        
        # Add new entry
        new_row = {
            'exp_id': 3,
            'exp_name': 'Random_Baseline',
            'strategy': 'random_assign',
            'completed_tasks': summary['completed_tasks'],
            'total_tasks': len(sampled_tasks),
            'jains_fairness_index': summary['jains_fairness_index'],
            'gini_coefficient': summary['gini_coefficient'],
            'mean_wait_time_minutes': summary['mean_wait_time_minutes'],
            'p90_wait_time_minutes': summary['p90_wait_time_minutes'],
            'mean_worker_utilization': summary['mean_worker_utilization'],
            'ewma_final_mean': summary.get('ewma_final_mean', None)
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.sort_values('exp_id').reset_index(drop=True)
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Updated: experiment_015_aggregate_results.csv")
        print(f"   Total rows: {len(df)}")
    
    print("\n" + "=" * 80)
    print("✅ RANDOM BASELINE RERUN COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  JFI: {summary['jains_fairness_index']:.3f}")
    print(f"  Mean Wait: {summary['mean_wait_time_minutes']:.2f} min")
    print(f"  Completed: {summary['completed_tasks']:,}/{len(sampled_tasks):,} tasks")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

