#!/usr/bin/env python3
"""
Experiment 016: Starvation Weight (λ₂) Interaction Analysis
Tests if optimal λ₂=0.5 remains consistent across different λ₁/λ₃ configurations.

Total: 28 simulations (~3.3 hours)
- 3 Baselines (Greedy, LAF, EWMA-Only)
- 25 Starvation Sweep (5 configs × 5 λ₂ values)
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

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "exp_016_starvation_weight_interaction"
EXPERIMENT_ID = "016"
DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(DATA_OUTPUT_DIR, f"exp_{EXPERIMENT_ID}_{TIMESTAMP}")

# Create output directory
os.makedirs(RUN_DIR, exist_ok=True)

# Dataset parameters (validated from Exp 012)
NUM_WORKERS = 4000
NUM_TASKS = 20000

# Robust baseline parameters (validated from prior experiments)
ROBUST_PARAMS = {
    'soft_threshold': 0.0,      # Disabled (Exp 011)
    'normalize_scores': True,   # Validated (Exp 008)
    'k': 15,                    # Standard
    'gamma': 0.5,               # Default (Exp 014)
}

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

def build_experiments():
    """Build all 28 experiment configurations."""
    experiments = []
    exp_id = 1
    
    # ========================================================================
    # GROUP 1: BASELINES (3 simulations)
    # ========================================================================
    baselines = [
        {'name': 'Greedy_Baseline', 'strategy': 'greedy', 'params': {}},
        {'name': 'LAF_Baseline', 'strategy': 'laf', 'params': {}},
        {'name': 'EWMA_Only_Baseline', 'strategy': 'ewma_only', 'params': {'gamma': 0.5}},
    ]
    
    for baseline in baselines:
        experiments.append({
            'exp_id': exp_id,
            'exp_name': baseline['name'],
            'strategy': baseline['strategy'],
            'params': baseline['params']
        })
        exp_id += 1
    
    # ========================================================================
    # GROUP 2: STARVATION WEIGHT SWEEP (25 simulations)
    # ========================================================================
    
    # 5 representative configurations (λ₁, λ₃, config_name)
    # Identified from Exp 014/015 Pareto frontier analysis
    configs = [
        (4.5, 0.5, "Best_JFI"),           # Maximum fairness
        (3.5, 1.0, "Balanced"),           # Standard balanced config
        (3.5, 2.5, "Mid_Range"),          # Near optimal balance score
        (2.5, 2.0, "Efficiency_Leaning"), # Efficiency-focused
        (2.5, 2.5, "Best_Efficiency")     # Maximum efficiency
    ]
    
    # 5 λ₂ values to test
    lambda2_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Generate all 25 combinations
    for (l1, l3, config_name) in configs:
        for l2 in lambda2_values:
            experiments.append({
                'exp_id': exp_id,
                'exp_name': f'Starvation_{config_name}_L1_{l1}_L3_{l3}_L2_{l2}',
                'strategy': 'composite',
                'params': {
                    'fairness_weight': l1,
                    'starvation_weight': l2,
                    'utility_weight': l3,
                    **ROBUST_PARAMS
                }
            })
            exp_id += 1
    
    return experiments

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_sample_data():
    """Load and sample workers/tasks using stratified temporal sampling."""
    print("\n" + "="*80)
    print("LOADING DATA...")
    print("="*80 + "\n")
    
    print("[STEP 1] Loading 3-hour peak dataset...")
    
    # Load from didi data directory
    data_path = project_root / "data" / "didi"
    all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
    
    print(f"\n[STEP 2] Time Ranges:")
    task_times = [t.release_time for t in all_tasks]
    worker_start = min(w.release_time for w in all_workers)
    worker_end = max(w.deadline for w in all_workers)
    
    print(f"  Task window: {min(task_times)} to {max(task_times)}")
    print(f"  Worker window: {worker_start} to {worker_end}")
    
    overlap_start = max(min(task_times), worker_start)
    overlap_end = min(max(task_times), worker_end)
    print(f"\n  ✅ Overlap window: {overlap_start} to {overlap_end}")
    
    duration_hours = (overlap_end - overlap_start).total_seconds() / 3600
    print(f"     Duration: {duration_hours:.2f} hours")
    
    print(f"\n[STEP 3] Stratified Temporal Sampling...")
    
    # Sample workers
    print(f"\n[STEP 3.1] Sampling {NUM_WORKERS} workers...")
    workers_df = pd.DataFrame([
        {'worker': w, 'release_time': w.release_time}
        for w in all_workers
    ])
    
    # Create time bins for stratification
    workers_df['time_bin'] = pd.cut(
        workers_df['release_time'].astype(int) // 10**9,
        bins=10,
        labels=False
    )
    
    # Stratified sampling
    sampled_workers_df = workers_df.groupby('time_bin', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), NUM_WORKERS // 10), random_state=42)
    )
    
    workers = [row['worker'] for _, row in sampled_workers_df.iterrows()]
    print(f"  ✅ Sampled {len(workers)} workers")
    
    # Sample tasks
    print(f"\n[STEP 3.2] Sampling {NUM_TASKS} tasks...")
    tasks_df = pd.DataFrame([
        {'task': t, 'release_time': t.release_time}
        for t in all_tasks
    ])
    
    tasks_df['time_bin'] = pd.cut(
        tasks_df['release_time'].astype(int) // 10**9,
        bins=10,
        labels=False
    )
    
    sampled_tasks_df = tasks_df.groupby('time_bin', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), NUM_TASKS // 10), random_state=42)
    )
    
    sampled_tasks = [row['task'] for _, row in sampled_tasks_df.iterrows()]
    print(f"  ✅ Sampled {len(sampled_tasks)} tasks")
    
    print("\n" + "="*80)
    print(f"✅ Loaded {len(workers)} workers and {len(sampled_tasks)} tasks")
    print("="*80 + "\n")
    
    return workers, sampled_tasks

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

def run_experiments():
    """Execute all experiment configurations."""
    
    # Load data
    workers, sampled_tasks = load_and_sample_data()
    
    # Build experiment configurations
    experiments = build_experiments()
    
    print(f"🧪 Running {len(experiments)} experiments...\n")
    
    # Store results
    results = []
    
    # Run each experiment
    for exp in experiments:
        print(f"🎲 Experiment {exp['exp_id']:03d}/{len(experiments):03d} - {exp['exp_name']}")
        print(f"   Strategy: {exp['strategy']}")
        print(f"   Params: {exp['params']}")
        
        # Deep copy workers and tasks to prevent mutation
        exp_workers = copy.deepcopy(workers)
        exp_tasks = copy.deepcopy(sampled_tasks)
        
        # CRITICAL: Update worker gamma if specified in config
        if 'gamma' in exp['params']:
            for worker in exp_workers:
                worker.gamma = exp['params']['gamma']
        
        # Create config for simulation
        sim_config = {
            'assignment_strategy': exp['strategy'],
            'strategy_params': exp['params']
        }
        
        # Run simulation
        summary = run_simulation(exp_workers, exp_tasks, sim_config=sim_config)
        
        # Add experiment metadata to summary
        summary['experiment'] = {
            'exp_id': exp['exp_id'],
            'exp_name': exp['exp_name'],
            'strategy': exp['strategy'],
            **exp['params']
        }
        
        # Save individual simulation result
        output_file = os.path.join(
            RUN_DIR, 
            f"exp_{exp['exp_id']:03d}_{exp['exp_name']}_summary.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump({'experiment': summary['experiment'], 'full_summary': summary}, 
                     f, indent=2, default=str)
        
        # Extract key metrics for aggregate results
        result_row = {
            'exp_id': exp['exp_id'],
            'exp_name': exp['exp_name'],
            'strategy': exp['strategy'],
            **exp['params'],
            'completed_tasks': summary.get('completed_tasks', 0),
            'task_assignment_ratio': summary.get('completed_tasks', 0) / len(sampled_tasks),
            'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
            'tasks_per_worker_gini': summary.get('final_gini_coefficient', 0),
            'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
            'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
            'mean_worker_utilization': summary.get('mean_worker_utilization', 0),
            'pct_workers_zero_tasks': summary.get('pct_workers_zero_tasks', 0),
            'tasks_per_worker_p10': summary.get('tasks_per_worker_p10', 0),
            'tasks_per_worker_p50': summary.get('tasks_per_worker_p50', 0),
            'tasks_per_worker_p90': summary.get('tasks_per_worker_p90', 0),
            'tasks_per_worker_mean': summary.get('tasks_per_worker_mean', 0),
            'tasks_per_worker_std': summary.get('tasks_per_worker_std', 0),
            'tasks_per_worker_cv': summary.get('tasks_per_worker_cv', 0),
            'wait_time_p10': summary.get('wait_time_p10', 0),
            'wait_time_p50': summary.get('wait_time_p50', 0),
            'wait_time_p95': summary.get('wait_time_p95', 0),
            'wait_time_mean': summary.get('wait_time_mean', 0),
            'wait_time_std': summary.get('wait_time_std', 0),
            'worker_util_p10': summary.get('worker_util_p10', 0),
            'worker_util_mean': summary.get('worker_util_mean', 0),
            'worker_util_p90': summary.get('worker_util_p90', 0),
            'worker_util_std': summary.get('worker_util_std', 0),
            'idle_time_p10': summary.get('idle_time_p10', 0),
            'idle_time_p50': summary.get('idle_time_p50', 0),
            'idle_time_p90': summary.get('idle_time_p90', 0),
            'idle_time_mean': summary.get('idle_time_mean', 0),
            'idle_time_std': summary.get('idle_time_std', 0),
            'runtime_seconds': summary.get('simulation_time_seconds', 0),
            'total_tasks': len(sampled_tasks)
        }
        
        results.append(result_row)
        
        # Print quick summary
        tar = result_row['task_assignment_ratio'] * 100
        jfi = result_row['jains_fairness_index']
        wait = result_row['mean_task_wait_time_min']
        
        print(f"   ✅ Completed: {result_row['completed_tasks']:,}/{len(sampled_tasks):,} tasks")
        print(f"   📊 JFI: {jfi:.3f}, Wait: {wait:.1f} min, TAR: {tar:.1f}%")
        print()
    
    # Save aggregate results
    aggregate_df = pd.DataFrame(results)
    aggregate_csv = os.path.join(DATA_OUTPUT_DIR, f"experiment_{EXPERIMENT_ID}_aggregate_results.csv")
    aggregate_df.to_csv(aggregate_csv, index=False)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\n📊 Results Summary:")
    print(f"   Total Simulations: {len(experiments)}")
    print(f"   Successful: {len(results)}")
    print(f"   Failed: {len(experiments) - len(results)}")
    print(f"\n📁 Output Files:")
    print(f"   Run directory: {RUN_DIR}")
    print(f"   Aggregate CSV: {aggregate_csv}")
    print(f"\n✅ Experiment 016 complete!")
    
    return aggregate_df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXPERIMENT 016: STARVATION WEIGHT INTERACTION ANALYSIS")
    print("="*80)
    print(f"Output Directory: {RUN_DIR}")
    print("="*80 + "\n")
    
    # Run all experiments
    results_df = run_experiments()
    
    print("\n🎉 All experiments completed successfully!")

