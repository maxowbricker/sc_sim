#!/usr/bin/env python3
"""
Experiment 011: Scalability Analysis
=====================================

Tests how system performance and fairness scale with worker population size.

Worker Counts: 2.5K, 5K, 10K, 15K
Configurations: 2 Pareto-efficient setups
Total: 8 experiments (~3-3.5 hours)
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_simulation_config, STRATEGY_PARAMS
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

def run_experiment_011():
    """Run Experiment 011: Scalability Analysis."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("EXPERIMENT 011: Scalability Analysis (REVISED)")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[OBJECTIVE] Analyze performance scaling with worker count")
    print("[FIXED WORKLOAD] 20,000 tasks per experiment")
    print("[WORKER COUNTS] 2K, 4K, 6K, 8K, 10K, 12K, 15K")
    print("[CONFIGURATION] Balanced (λ₁=2.0, λ₂=0.8, λ₃=1.0, θ=0.5)")
    print("[TOTAL EXPERIMENTS] 7")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / f"exp_011_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {output_dir}")
    print()
    
    # Fixed settings for all experiments
    FIXED_SETTINGS = {
        'normalize_scores': True,
        'enable_diagnostics': False,
        'gamma': 0.5,
        'k': 15
    }
    
    # Define experiment configurations
    experiments = []
    exp_id = 1
    
    # 7 different worker counts with FIXED 20K tasks
    worker_counts = [2000, 4000, 6000, 8000, 10000, 12000, 15000]
    FIXED_TASK_COUNT = 20000
    
    # Single Balanced configuration
    config = {
        'name': 'Balanced',
        'fairness_weight': 2.0,
        'starvation_weight': 0.8,
        'utility_weight': 1.0,
        'soft_threshold': 0.0  # Disabled to test if threshold is causing failures
    }
    
    for worker_count in worker_counts:
        experiments.append({
            'id': exp_id,
            'name': f"{worker_count // 1000}K_workers",
            'description': f"{worker_count:,} workers, 20K tasks",
            'worker_count': worker_count,
            'task_count': FIXED_TASK_COUNT,
            'config_name': config['name'],
            **config,
            **FIXED_SETTINGS
        })
        exp_id += 1
    
    print(f"[EXPERIMENTS CONFIGURED] {len(experiments)}")
    print()
    
    # Display experiment matrix
    print("Experiment Matrix:")
    print("-" * 80)
    print(f"{'ID':<4} {'Workers':<10} {'Tasks':<10} {'Tasks/Worker':<15}")
    print("-" * 80)
    for exp in experiments:
        tasks_per_worker = exp['task_count'] / exp['worker_count']
        print(f"{exp['id']:<4} {exp['worker_count']:<10,} {exp['task_count']:<10,} {tasks_per_worker:<15.1f}")
    print("-" * 80)
    print(f"Config: λ₁={config['fairness_weight']}, λ₂={config['starvation_weight']}, "
          f"λ₃={config['utility_weight']}, θ={config['soft_threshold']}")
    print("-" * 80)
    print()
    
    # Load full dataset once
    print("Loading full dataset...")
    data_path = project_root / "data" / "didi"
    all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
    print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")
    
    # Convert to lists if needed (for sampling)
    if not isinstance(all_workers, list):
        all_workers = list(all_workers)
    if not isinstance(all_tasks, list):
        all_tasks = list(all_tasks)
    
    # Sample 20K tasks once (used for all experiments)
    import random
    random.seed(42)
    sampled_tasks = random.sample(all_tasks, min(FIXED_TASK_COUNT, len(all_tasks)))
    print(f"✅ Sampled {len(sampled_tasks):,} tasks for experiments")
    print()
    
    # Run experiments
    results = []
    successful_runs = 0
    
    for i, exp in enumerate(experiments, 1):
        exp_start = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp['id']}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")
        print(f"Workers: {exp['worker_count']:,}")
        print(f"Config: {exp['config_name']}")
        print(f"λ₁={exp['fairness_weight']}, λ₂={exp['starvation_weight']}, "
              f"λ₃={exp['utility_weight']}, θ={exp['soft_threshold']}")
        print()
        
        try:
            # Sample workers for this experiment (using same seed for all = nested subsets)
            random.seed(42)  # Same seed = 2K workers is subset of 4K workers, etc.
            sampled_workers = random.sample(all_workers, min(exp['worker_count'], len(all_workers)))
            
            print(f"Sampled {len(sampled_workers):,} workers")
            print(f"Running simulation with {len(sampled_tasks):,} tasks...")
            print(f"  → {len(sampled_tasks) / len(sampled_workers):.1f} tasks per worker")
            
            # Update strategy params
            STRATEGY_PARAMS['composite']['fairness_weight'] = exp['fairness_weight']
            STRATEGY_PARAMS['composite']['starvation_weight'] = exp['starvation_weight']
            STRATEGY_PARAMS['composite']['utility_weight'] = exp['utility_weight']
            STRATEGY_PARAMS['composite']['soft_threshold'] = exp['soft_threshold']
            STRATEGY_PARAMS['composite']['normalize_scores'] = exp['normalize_scores']
            STRATEGY_PARAMS['composite']['gamma'] = exp['gamma']
            STRATEGY_PARAMS['composite']['enable_diagnostics'] = exp['enable_diagnostics']
            
            # Run simulation with FIXED 20K tasks
            cfg = get_simulation_config()
            summary = run_simulation(sampled_workers, sampled_tasks, sim_config=cfg)
            
            exp_end = datetime.now()
            exp_duration = (exp_end - exp_start).total_seconds()
            
            # Calculate TAR (not in summary dict from run_simulation)
            completed = summary.get('completed_tasks', 0)
            tar = completed / exp['task_count'] if exp['task_count'] > 0 else 0
            jfi = summary.get('final_jains_fairness_index', 0)
            
            print(f"\n✅ Experiment {exp['id']} complete in {exp_duration/60:.1f} minutes")
            print(f"   Completed: {completed:,} tasks")
            print(f"   TAR: {tar:.2%}")
            print(f"   JFI: {jfi:.3f}")
            print(f"   Gini: {summary.get('tasks_per_worker_gini', 0):.3f}")
            print(f"   Mean Wait: {summary.get('avg_wait_time_minutes', 0):.1f} min")
            
            # Store results with all 78+ metrics
            result = {
                'experiment_id': exp['id'],
                'name': exp['name'],
                'description': exp['description'],
                'worker_count': exp['worker_count'],
                'task_count': exp['task_count'],
                'tasks_per_worker_ratio': exp['task_count'] / exp['worker_count'],
                'config_name': exp['config_name'],
                
                # Configuration
                'fairness_weight': exp['fairness_weight'],
                'starvation_weight': exp['starvation_weight'],
                'utility_weight': exp['utility_weight'],
                'soft_threshold': exp['soft_threshold'],
                'normalize_scores': exp['normalize_scores'],
                'gamma': exp['gamma'],
                
                # Core metrics
                'completed_tasks': completed,
                'task_assignment_ratio': tar,
                'jains_fairness_index': jfi,
                
                # Task wait time distribution
                'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
                'std_task_wait_time_min': summary.get('std_wait_time_minutes', 0),
                'p90_task_wait_time_min': summary.get('p90_wait_time_minutes', 0),
                'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
                'max_task_wait_time_min': summary.get('max_wait_time_minutes', 0),
                'cv_task_wait_time': summary.get('cv_wait_time', 0),
                
                # Worker idle time distribution
                'mean_worker_idle_time_min': summary.get('mean_worker_idle_time_min', 0),
                'std_worker_idle_time_min': summary.get('std_worker_idle_time_min', 0),
                'p90_worker_idle_time_min': summary.get('p90_worker_idle_time_min', 0),
                'max_worker_idle_time_min': summary.get('max_worker_idle_time_min', 0),
                'cv_worker_idle_time': summary.get('cv_worker_idle_time', 0),
                
                # Worker task distribution (NEW v2.0 metrics!)
                'tasks_per_worker_mean': summary.get('tasks_per_worker_mean', 0),
                'tasks_per_worker_std': summary.get('tasks_per_worker_std', 0),
                'tasks_per_worker_cv': summary.get('tasks_per_worker_cv', 0),
                'tasks_per_worker_gini': summary.get('tasks_per_worker_gini', 0),
                'tasks_per_worker_p10': summary.get('tasks_per_worker_p10', 0),
                'tasks_per_worker_p50': summary.get('tasks_per_worker_p50', 0),
                'tasks_per_worker_p90': summary.get('tasks_per_worker_p90', 0),
                'pct_workers_zero_tasks': summary.get('pct_workers_zero_tasks', 0),
                'pct_workers_single_task': summary.get('pct_workers_single_task', 0),
                
                # Worker utilization (NEW v2.0 metrics!)
                'mean_worker_utilization': summary.get('mean_worker_utilization', 0),
                'std_worker_utilization': summary.get('std_worker_utilization', 0),
                'p10_worker_utilization': summary.get('p10_worker_utilization', 0),
                'p90_worker_utilization': summary.get('p90_worker_utilization', 0),
                
                # Pickup distance distribution (NEW v2.0 metrics!)
                'mean_pickup_distance_km': summary.get('avg_pickup_distance_km', 0),
                'std_pickup_distance_km': summary.get('std_pickup_distance_km', 0),
                'p90_pickup_distance_km': summary.get('p90_pickup_distance_km', 0),
                'max_pickup_distance_km': summary.get('max_pickup_distance_km', 0),
                
                # Task deferrals (NEW v2.0 metrics!)
                'total_deferrals': summary.get('total_deferrals', 0),
                'pct_tasks_deferred': summary.get('pct_tasks_deferred', 0),
                'mean_deferrals_per_task': summary.get('mean_deferrals_per_task', 0),
                'max_deferrals_per_task': summary.get('max_deferrals_per_task', 0),
                
                # Assignment timing (NEW v2.0 metrics!)
                'mean_assignment_delay_sec': summary.get('mean_assignment_delay_sec', 0),
                'std_assignment_delay_sec': summary.get('std_assignment_delay_sec', 0),
                'p90_assignment_delay_sec': summary.get('p90_assignment_delay_sec', 0),
                
                # System metrics
                'total_travel_km': summary.get('total_travel_distance_km', 0),
                'empty_km_ratio': summary.get('empty_km_share', 0),
                'peak_backlog': summary.get('backlog_peak', 0),
                'ewma_cv': summary.get('ewma_cv', 0),
                
                # Metadata
                'duration_seconds': exp_duration,
                'timestamp': exp_end.isoformat()
            }
            
            results.append(result)
            successful_runs += 1
            
            # Save individual result
            result_file = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_summary.json"
            with open(result_file, 'w') as f:
                full_result = {
                    **result,
                    'full_summary': {k: v for k, v in summary.items() 
                                   if k not in ['metric_tracker', 'diagnostic_tracker', 'workers_df']}
                }
                json.dump(full_result, f, indent=2, default=str)
            
            print(f"   💾 Saved to {result_file.name}")
            
            # Progress update
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time_per_exp = elapsed / i
            remaining_exps = len(experiments) - i
            eta_seconds = remaining_exps * avg_time_per_exp
            eta = datetime.now() + pd.Timedelta(seconds=eta_seconds)
            
            print(f"\n📊 Progress: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f} min")
            print(f"   Avg per exp: {avg_time_per_exp/60:.1f} min")
            print(f"   ETA: {eta.strftime('%H:%M:%S')} (~{eta_seconds/60:.0f} min remaining)")
            
        except Exception as e:
            print(f"\n❌ Experiment {exp['id']} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregate results
    if results:
        df = pd.DataFrame(results)
        csv_path = output_dir.parent / "experiment_011_aggregate_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Aggregate results saved to {csv_path}")
    
    # Final summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 011 COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {successful_runs}/{len(experiments)}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average per experiment: {total_duration/len(experiments)/60:.1f} minutes")
    print(f"\nOutput directory: {output_dir}")
    print(f"Aggregate CSV: {csv_path if results else 'N/A'}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_011()

