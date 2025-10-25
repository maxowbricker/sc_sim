#!/usr/bin/env python3
"""
Standalone Baseline Runner for Experiment 011
==============================================

This script runs ALL 4 baselines (Greedy, LAF, EWMA-Only, Random, FATP-ANN) 
for ALL 7 worker counts in Experiment 011.

IMPORTANT:
- Does NOT modify the original CSV
- Saves results to a separate "baselines_exp011_results.csv"
- Saves individual JSON files for each run
- You manually review and merge in the morning

Total Simulations: 28 (4 baselines × 7 worker counts)
Estimated Runtime: ~7-8 hours

Output:
- Individual JSONs: baselines/exp_XXX_{baseline}_{workers}_summary.json
- Summary CSV: baselines_exp011_results.csv
"""

import sys
import json
import copy
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

# ============================================================================
# BASELINE CONFIGURATIONS
# ============================================================================

BASELINES = {
    'greedy': {
        'name': 'Greedy',
        'assignment_strategy': 'greedy',
        'strategy_params': {}
    },
    'laf': {
        'name': 'LAF',
        'assignment_strategy': 'laf',
        'strategy_params': {}
    },
    'ewma_only': {
        'name': 'EWMA_Only',
        'assignment_strategy': 'ewma_only',
        'strategy_params': {'gamma': 0.5}
    },
    'random_assign': {
        'name': 'Random',
        'assignment_strategy': 'random_assign',
        'strategy_params': {'k': 15}
    },
    'fatp_ann': {
        'name': 'FATP_ANN',
        'assignment_strategy': 'fatp_ann',
        'strategy_params': {
            'mu': 0.5,
            'alpha_scale': 0.5,
            'use_k_nearest': False,
            'k': 15
        }
    }
}

# Worker counts from original exp_011
WORKER_COUNTS = [2000, 4000, 6000, 8000, 10000, 12000, 15000]
FIXED_TASK_COUNT = 20000

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_sample_data(target_workers, target_tasks):
    """
    Load and sample data to match experiment specifications.
    Uses stratified temporal sampling for consistency.
    """
    print(f"   Loading data: {target_workers:,} workers, {target_tasks:,} tasks")
    
    # Load full dataset
    data_path = project_root / "data" / "didi"
    workers, tasks = load_workers_tasks('didi', str(data_path))
    
    # Sample workers
    if len(workers) > target_workers:
        workers_df = pd.DataFrame([{
            'worker': w,
            'release_time': w.release_time
        } for w in workers])
        
        workers_df['time_bin'] = pd.cut(
            (workers_df['release_time'] - workers_df['release_time'].min()).dt.total_seconds(),
            bins=50,
            labels=False
        )
        
        sampled_workers_df = workers_df.groupby('time_bin', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, int(target_workers / 50))), random_state=42)
        )
        workers = sampled_workers_df['worker'].tolist()[:target_workers]
    
    # Sample tasks
    if len(tasks) > target_tasks:
        tasks_df = pd.DataFrame([{
            'task': t,
            'release_time': t.release_time
        } for t in tasks])
        
        tasks_df['time_bin'] = pd.cut(
            (tasks_df['release_time'] - tasks_df['release_time'].min()).dt.total_seconds(),
            bins=50,
            labels=False
        )
        
        sampled_tasks_df = tasks_df.groupby('time_bin', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, int(target_tasks / 50))), random_state=42)
        )
        tasks = sampled_tasks_df['task'].tolist()[:target_tasks]
    
    print(f"   ✅ Loaded: {len(workers):,} workers, {len(tasks):,} tasks")
    return workers, tasks

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_single_baseline(baseline_key, baseline_config, workers, tasks, worker_count, run_id, output_dir):
    """
    Run a single baseline simulation and save results.
    """
    worker_label = f"{worker_count // 1000}K"
    tasks_per_worker = FIXED_TASK_COUNT / worker_count
    
    print(f"   Strategy: {baseline_config['assignment_strategy']}")
    print(f"   Workers: {worker_count:,} | Tasks: {FIXED_TASK_COUNT:,} | Ratio: {tasks_per_worker:.1f}")
    
    # Deep copy for isolation
    sim_workers = copy.deepcopy(workers)
    sim_tasks = copy.deepcopy(tasks)
    
    # Configure simulation
    sim_config = {
        'assignment_strategy': baseline_config['assignment_strategy'],
        'strategy_params': baseline_config['strategy_params']
    }
    
    # Run simulation
    start_time = datetime.now()
    print(f"   ⏱️  Started: {start_time.strftime('%H:%M:%S')}")
    
    try:
        summary = run_simulation(sim_workers, sim_tasks, sim_config=sim_config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"   ⏱️  Completed: {end_time.strftime('%H:%M:%S')} (Duration: {duration:.1f} min)")
        
        # Extract key metrics
        result = {
            'run_id': run_id,
            'baseline': baseline_key,
            'baseline_name': baseline_config['name'],
            'worker_count': worker_count,
            'task_count': FIXED_TASK_COUNT,
            'tasks_per_worker_ratio': tasks_per_worker,
            'duration_min': duration,
            
            # Core metrics
            'completed_tasks': summary.get('completed_tasks', 0),
            'total_tasks': summary.get('total_tasks', FIXED_TASK_COUNT),
            'task_assignment_ratio': summary.get('completed_tasks', 0) / FIXED_TASK_COUNT,
            
            # Fairness
            'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
            'tasks_per_worker_gini': summary.get('tasks_per_worker_gini', 0),
            'tasks_per_worker_mean': summary.get('tasks_per_worker_mean', 0),
            'tasks_per_worker_std': summary.get('tasks_per_worker_std', 0),
            'tasks_per_worker_cv': summary.get('tasks_per_worker_cv', 0),
            'pct_workers_zero_tasks': summary.get('pct_workers_zero_tasks', 0),
            
            # Efficiency
            'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
            'std_task_wait_time_min': summary.get('std_wait_time_minutes', 0),
            'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
            'max_task_wait_time_min': summary.get('max_wait_time_minutes', 0),
            
            # Worker utilization
            'mean_worker_utilization': summary.get('mean_worker_utilization', 0),
            'std_worker_utilization': summary.get('std_worker_utilization', 0),
            
            # Travel
            'total_travel_km': summary.get('total_travel_km', 0),
            'empty_km': summary.get('empty_km', 0),
            'empty_km_ratio': summary.get('empty_km', 0) / summary.get('total_travel_km', 1) if summary.get('total_travel_km', 0) > 0 else 0,
            
            # Timestamps
            'timestamp': end_time.isoformat()
        }
        
        # Save individual JSON
        json_filename = f"exp_{run_id:03d}_{baseline_config['name']}_{worker_label}_summary.json"
        json_path = output_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   💾 Saved: {json_filename}")
        
        # Print key results
        print(f"   📊 JFI: {result['jains_fairness_index']:.4f} | "
              f"TAR: {result['task_assignment_ratio']:.2%} | "
              f"Wait: {result['mean_task_wait_time_min']:.2f} min")
        
        return result
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 011: BASELINE SCALABILITY ANALYSIS")
    print("=" * 80)
    print()
    print("This will run ALL baselines for ALL worker counts:")
    print(f"  Baselines: {', '.join(BASELINES.keys())}")
    print(f"  Worker counts: {', '.join(f'{w//1000}K' for w in WORKER_COUNTS)}")
    print(f"  Total simulations: {len(BASELINES) * len(WORKER_COUNTS)}")
    print(f"  Fixed task count: {FIXED_TASK_COUNT:,}")
    print()
    print("Estimated runtime: 7-8 hours")
    print()
    print("OUTPUT:")
    print("  - Individual JSONs: baselines/exp_XXX_{baseline}_{workers}_summary.json")
    print("  - Summary CSV: baselines_exp011_results.csv")
    print("  - ORIGINAL CSV: WILL NOT BE MODIFIED")
    print()
    
    input("Press Enter to start, or Ctrl+C to cancel...")
    print()
    
    # Create output directory
    exp_dir = Path(__file__).parent
    output_dir = exp_dir / "baselines"
    output_dir.mkdir(exist_ok=True)
    
    # Track results
    all_results = []
    run_id = 1
    total_sims = len(BASELINES) * len(WORKER_COUNTS)
    current_sim = 0
    
    overall_start = datetime.now()
    
    # Run all combinations
    for baseline_key, baseline_config in BASELINES.items():
        for worker_count in WORKER_COUNTS:
            current_sim += 1
            
            print("=" * 80)
            print(f"SIMULATION {current_sim}/{total_sims}: {baseline_config['name']} with {worker_count:,} workers")
            print("=" * 80)
            print()
            
            # Load data for this configuration
            workers, tasks = load_and_sample_data(worker_count, FIXED_TASK_COUNT)
            
            # Run simulation
            result = run_single_baseline(
                baseline_key, baseline_config, 
                workers, tasks, 
                worker_count, run_id, 
                output_dir
            )
            
            if result:
                all_results.append(result)
                run_id += 1
            
            print()
            
            # Progress update
            elapsed = (datetime.now() - overall_start).total_seconds() / 60
            avg_per_sim = elapsed / current_sim
            remaining = (total_sims - current_sim) * avg_per_sim
            print(f"📊 PROGRESS: {current_sim}/{total_sims} complete")
            print(f"   Elapsed: {elapsed:.1f} min | Remaining: {remaining:.1f} min (~{remaining/60:.1f} hours)")
            print()
    
    # Save aggregate CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = exp_dir / "baselines_exp011_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        print("=" * 80)
        print("✅ ALL SIMULATIONS COMPLETE")
        print("=" * 80)
        print()
        print(f"📊 Results saved:")
        print(f"   CSV: {csv_path.name}")
        print(f"   JSONs: baselines/ directory ({len(all_results)} files)")
        print()
        print(f"⏱️  Total runtime: {(datetime.now() - overall_start).total_seconds() / 60:.1f} minutes")
        print()
        print("NEXT STEPS:")
        print("  1. Review baselines_exp011_results.csv")
        print("  2. Compare with original experiment_011_aggregate_results.csv")
        print("  3. Manually merge if satisfied with results")
        print("  4. Update analysis.ipynb to include baselines")
        print()
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()

