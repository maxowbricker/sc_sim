#!/usr/bin/env python3
"""
Experiment 015: EWMA Temporal & Baseline Validation
Collects temporal EWMA data and validates Random baseline.

Total: 33 simulations (~3.85 hours)
- 3 Baselines (Greedy, LAF, Random)
- 1 EWMA-Only
- 25 Pareto Sweep (5×5 grid)
- 4 Gamma Sensitivity
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

EXPERIMENT_NAME = "exp_015_temporal_ewma_validation"
EXPERIMENT_ID = "015"
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
    'starvation_weight': 0.5,   # Validated (Exp 009)
    'normalize_scores': True,   # Validated (Exp 008)
    'k': 15,                    # Standard
}

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

def build_experiments():
    """Build all 33 experiment configurations."""
    experiments = []
    exp_id = 1
    
    # ========================================================================
    # GROUP 1: BASELINES (3 simulations)
    # ========================================================================
    baselines = [
        {'name': 'Greedy_Baseline', 'strategy': 'greedy', 'params': {}},
        {'name': 'LAF_Baseline', 'strategy': 'laf', 'params': {}},
        {'name': 'Random_Baseline', 'strategy': 'random_assign', 'params': {}},
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
    # GROUP 2: EWMA-ONLY (1 simulation)
    # ========================================================================
    experiments.append({
        'exp_id': exp_id,
        'exp_name': 'EWMA_Only_G_0.5',
        'strategy': 'ewma_only',
        'params': {'gamma': 0.5}
    })
    exp_id += 1
    
    # ========================================================================
    # GROUP 3: PARETO SWEEP (25 simulations)
    # ========================================================================
    lambda1_values = [2.5, 3.0, 3.5, 4.0, 4.5]
    lambda3_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    for l1 in lambda1_values:
        for l3 in lambda3_values:
            experiments.append({
                'exp_id': exp_id,
                'exp_name': f'Pareto_L1_{l1:.1f}_L3_{l3:.1f}',
                'strategy': 'composite',
                'params': {
                    **ROBUST_PARAMS,
                    'fairness_weight': l1,
                    'utility_weight': l3,
                    'gamma': 0.5,
                }
            })
            exp_id += 1
    
    # ========================================================================
    # GROUP 4: GAMMA SENSITIVITY (4 simulations)
    # ========================================================================
    # Balanced configuration: λ₁=3.5, λ₃=1.0
    gamma_values = [0.1, 0.3, 0.7, 0.9]  # γ=0.5 already in Pareto sweep
    
    for g in gamma_values:
        experiments.append({
            'exp_id': exp_id,
            'exp_name': f'Gamma_Balanced_G_{g:.1f}',
            'strategy': 'composite',
            'params': {
                **ROBUST_PARAMS,
                'fairness_weight': 3.5,
                'utility_weight': 1.0,
                'gamma': g,
            }
        })
        exp_id += 1
    
    return experiments

# ============================================================================
# DATA LOADING & SAMPLING
# ============================================================================

def load_and_sample_data():
    """Load and sample workers/tasks using stratified temporal sampling."""
    print("=" * 80)
    print("LOADING AND SAMPLING DATA")
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
    print(f"  Total tasks available: {len(all_tasks):,}")
    
    print(f"  Worker window: {worker_start} to {worker_end}")
    print(f"  Total workers available: {len(all_workers):,}")
    
    # Find overlap
    overlap_start = max(worker_start, task_start)
    overlap_end = min(worker_end, task_end)
    
    print(f"\n  ✅ Overlap window: {overlap_start} to {overlap_end}")
    duration_hours = (overlap_end - overlap_start).total_seconds() / 3600
    print(f"     Duration: {duration_hours:.2f} hours")
    
    # ========================================================================
    # STRATIFIED TEMPORAL SAMPLING
    # ========================================================================
    print(f"\n[STEP 3] Stratified Temporal Sampling...")
    
    # Sample workers with stratified approach
    print(f"\n[STEP 3.1] Sampling {NUM_WORKERS:,} workers...")
    
    # Create temporal bins (12 bins of 15 minutes each for 3-hour window)
    num_bins = 12
    bin_duration = pd.Timedelta(minutes=15)
    bin_edges = [overlap_start + i * bin_duration for i in range(num_bins + 1)]
    
    workers_per_bin = NUM_WORKERS // num_bins
    sampled_workers = []
    
    print(f"  Distribution across bins:")
    for i in range(num_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Workers available in this bin
        bin_workers = [w for w in all_workers 
                      if w.release_time <= bin_start and w.deadline >= bin_end]
        
        # Sample from this bin
        if len(bin_workers) >= workers_per_bin:
            sampled = pd.Series(bin_workers).sample(n=workers_per_bin, random_state=42).tolist()
        else:
            sampled = bin_workers
        
        sampled_workers.extend(sampled)
        
        print(f"    {bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}: "
              f"{len(sampled):4d} workers (from {len(bin_workers):5d} available)")
    
    # Ensure exactly NUM_WORKERS
    sampled_workers = sampled_workers[:NUM_WORKERS]
    print(f"  ✅ Sampled {len(sampled_workers):,} workers")
    print(f"  Coverage: {sampled_workers[0].release_time.strftime('%H:%M')} to "
          f"{max(w.deadline for w in sampled_workers).strftime('%H:%M')}")
    
    # Count workers available at task start
    first_task_time = overlap_start
    available_at_start = sum(1 for w in sampled_workers 
                            if w.release_time <= first_task_time <= w.deadline)
    print(f"  📊 Workers available at first task ({first_task_time.strftime('%H:%M')}): "
          f"{available_at_start} ({available_at_start/len(sampled_workers)*100:.1f}%)")
    
    # Sample tasks uniformly from overlap window
    print(f"\n[STEP 3.2] Sampling {NUM_TASKS:,} tasks...")
    overlap_tasks = [t for t in all_tasks 
                    if overlap_start <= t.release_time <= overlap_end]
    
    if len(overlap_tasks) >= NUM_TASKS:
        sampled_tasks = pd.Series(overlap_tasks).sample(n=NUM_TASKS, random_state=42).tolist()
    else:
        print(f"  ⚠️  Only {len(overlap_tasks)} tasks in overlap window")
        sampled_tasks = overlap_tasks
    
    print(f"  ✅ Sampled {len(sampled_tasks):,} tasks")
    
    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"✅ Tasks sampled: {len(sampled_tasks):,}")
    print(f"✅ Worker samples created: 1")
    print(f"\nWorker availability at first task:")
    print(f"   {NUM_WORKERS:,} workers: {available_at_start:5d} available "
          f"({available_at_start/NUM_WORKERS*100:5.1f}%)")
    print("=" * 80)
    
    return sampled_workers, sampled_tasks

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_experiments():
    """Run all experiments and save results."""
    # Load and sample data once
    workers, tasks = load_and_sample_data()
    
    print(f"\n✅ Loaded {len(workers):,} workers and {len(tasks):,} tasks")
    
    # Build experiment configurations
    experiments = build_experiments()
    
    print(f"\n📋 Total experiments: {len(experiments)}")
    print(f"   Group 1 (Baselines): 3 simulations")
    print(f"   Group 2 (EWMA-Only): 1 simulation")
    print(f"   Group 3 (Pareto Sweep): 25 simulations")
    print(f"   Group 4 (Gamma Sensitivity): 4 simulations")
    
    # Estimate runtime
    est_time_per_sim = 7  # minutes
    total_est_time = len(experiments) * est_time_per_sim
    print(f"\n⏱️  Estimated total runtime: ~{total_est_time} minutes (~{total_est_time/60:.1f} hours)")
    
    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS")
    print("=" * 80)
    
    results = []
    successful = 0
    failed = 0
    
    start_time = datetime.now()
    
    for exp in experiments:
        print(f"\n🔄 Experiment {exp['exp_id']:03d}/{len(experiments):03d} - {exp['exp_name']}")
        print(f"   Strategy: {exp['strategy']}")
        if exp['params']:
            params_str = ', '.join([f"{k}={v}" for k, v in exp['params'].items() 
                                   if k in ['fairness_weight', 'utility_weight', 'gamma']])
            if params_str:
                print(f"   {params_str}")
        
        try:
            # Deep copy workers and tasks to prevent mutation
            exp_workers = copy.deepcopy(workers)
            exp_tasks = copy.deepcopy(tasks)
            
            # CRITICAL: Update worker.gamma if specified in params
            if 'gamma' in exp['params']:
                for worker in exp_workers:
                    worker.gamma = exp['params']['gamma']
            
            # Configure simulation
            sim_config = {
                'assignment_strategy': exp['strategy'],
                'strategy_params': exp['params']
            }
            
            # Run simulation
            summary = run_simulation(exp_workers, exp_tasks, sim_config=sim_config)
            
            # Extract key metrics
            result = {
                'exp_id': exp['exp_id'],
                'exp_name': exp['exp_name'],
                'strategy': exp['strategy'],
                **{k: v for k, v in exp['params'].items() 
                   if k in ['fairness_weight', 'starvation_weight', 'utility_weight', 
                           'gamma', 'soft_threshold', 'normalize_scores', 'k']},
                'completed_tasks': summary.get('completed_tasks', 0),
                'task_assignment_ratio': summary.get('completed_tasks', 0) / len(tasks),
                'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                'tasks_per_worker_gini': summary.get('final_tasks_per_worker_gini', 0),
                'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
                'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
                'mean_worker_utilization': summary.get('final_mean_worker_utilization', 0),
                'pct_workers_zero_tasks': summary.get('final_pct_workers_zero_tasks', 0),
                'tasks_per_worker_p10': summary.get('final_tasks_per_worker_p10', 0),
                'tasks_per_worker_p50': summary.get('final_tasks_per_worker_p50', 0),
                'tasks_per_worker_p90': summary.get('final_tasks_per_worker_p90', 0),
                'tasks_per_worker_mean': summary.get('final_tasks_per_worker_mean', 0),
                'tasks_per_worker_std': summary.get('final_tasks_per_worker_std', 0),
                'tasks_per_worker_cv': summary.get('final_tasks_per_worker_cv', 0),
                'wait_time_p10': summary.get('wait_time_p10', 0),
                'wait_time_p50': summary.get('wait_time_p50', 0),
                'wait_time_p95': summary.get('p95_wait_time_minutes', 0),
                'wait_time_mean': summary.get('avg_wait_time_minutes', 0),
                'wait_time_std': summary.get('std_wait_time_minutes', 0),
                'worker_util_p10': summary.get('final_worker_util_p10', 0),
                'worker_util_mean': summary.get('final_mean_worker_utilization', 0),
                'worker_util_p90': summary.get('final_worker_util_p90', 0),
                'worker_util_std': summary.get('final_worker_util_std', 0),
                'idle_time_p10': summary.get('final_idle_time_p10', 0),
                'idle_time_p50': summary.get('final_idle_time_p50', 0),
                'idle_time_p90': summary.get('final_idle_time_p90', 0),
                'idle_time_mean': summary.get('final_idle_time_mean', 0),
                'idle_time_std': summary.get('final_idle_time_std', 0),
                'runtime_seconds': summary.get('runtime_seconds', 0),
                'ewma_final_mean': summary.get('ewma_final_mean', None),  # New for Exp 015
            }
            
            results.append(result)
            successful += 1
            
            # Save individual simulation summary (including temporal data)
            summary_file = os.path.join(RUN_DIR, f"exp_{exp['exp_id']:03d}_{exp['exp_name']}_summary.json")
            # Remove non-serializable objects
            summary_clean = {k: v for k, v in summary.items() 
                           if k not in ['metric_tracker', 'diagnostic_tracker']}
            with open(summary_file, 'w') as f:
                json.dump(summary_clean, f, indent=2, default=str)
            
            print(f"   ✅ Completed: {result['completed_tasks']:,}/{len(tasks):,} tasks "
                  f"(TAR: {result['task_assignment_ratio']*100:.1f}%)")
            print(f"   JFI: {result['jains_fairness_index']:.3f} | "
                  f"Gini: {result['tasks_per_worker_gini']:.3f} | "
                  f"Wait: {result['mean_task_wait_time_min']:.2f} min")
            print(f"   Runtime: {result['runtime_seconds']:.1f}s")
            if result['ewma_final_mean'] is not None:
                print(f"   EWMA Final: {result['ewma_final_mean']:.4f}")
            print(f"   💾 Saved: {os.path.basename(summary_file)}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            failed += 1
            continue
        
        # Progress update
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        completed = successful + failed
        if completed > 0:
            est_remaining = (elapsed / completed) * (len(experiments) - completed)
            print(f"\n   Progress: {completed}/{len(experiments)} | "
                  f"✅ {successful} | ❌ {failed} | "
                  f"Elapsed: {elapsed:.1f}m | Est. Remaining: {est_remaining:.1f}m")
    
    # Save aggregate results
    print("\n" + "=" * 80)
    print("SAVING AGGREGATE RESULTS")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    csv_file = os.path.join(DATA_OUTPUT_DIR, f"experiment_{EXPERIMENT_ID}_aggregate_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Saved aggregate CSV: {os.path.basename(csv_file)}")
    print(f"   Total rows: {len(df)}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Runtime: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    
    print(f"\n✅ Successful simulations: {successful}/{len(experiments)}")
    print(f"❌ Failed simulations: {failed}/{len(experiments)}")
    
    if successful > 0:
        print(f"\n📊 Success Metrics:")
        print(f"   Mean TAR: {df['task_assignment_ratio'].mean()*100:.1f}%")
        print(f"   Mean JFI: {df['jains_fairness_index'].mean():.3f}")
        print(f"   Mean Wait Time: {df['mean_task_wait_time_min'].mean():.2f} min")
    
    print(f"\n🎉 All experiments completed!")
    
    print(f"\n📁 Results saved to:")
    print(f"   - {csv_file}")
    print(f"   - {RUN_DIR}/")
    
    print("\n" + "=" * 80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {EXPERIMENT_ID}: EWMA TEMPORAL & BASELINE VALIDATION")
    print(f"{'='*80}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {RUN_DIR}")
    
    run_experiments()

