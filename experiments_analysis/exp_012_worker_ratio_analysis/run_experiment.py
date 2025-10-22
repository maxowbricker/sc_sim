#!/usr/bin/env python3
"""
Experiment 012: Worker-to-Task Ratio Analysis
==============================================

Identifies the optimal worker count for a fixed 20K task workload using
stratified temporal sampling to ensure worker availability aligns with
task arrivals.

Key Innovation: Stratified sampling across temporal bins addresses the
temporal misalignment issue discovered in Experiment 011.

Worker Counts: 2K, 3K, 4K, 5K, 6K, 7K, 8K, 9K, 10K, 12K, 15K
Configuration: λ₁=2.0, λ₂=0.5, λ₃=1.0, θ=0.0 (balanced + robust)
Total: 11 experiments (~2.5-3 hours)
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import copy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_simulation_config, STRATEGY_PARAMS
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample
from simulator.simulation import run_simulation


def run_experiment_012():
    """Run Experiment 012: Worker-to-Task Ratio Analysis."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("EXPERIMENT 012: Worker-to-Task Ratio Analysis")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[OBJECTIVE] Find optimal worker count for 20K task workload")
    print("[METHOD] Stratified temporal sampling (12 bins)")
    print("[CONFIGURATION] λ₁=2.0, λ₂=0.5, λ₃=1.0, θ=0.0")
    print("[WORKER COUNTS] 2K, 3K, 4K, 5K, 6K, 7K, 8K, 9K, 10K, 12K, 15K")
    print("[TOTAL EXPERIMENTS] 11")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / f"exp_012_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {output_dir}")
    print()
    
    # Fixed settings for all experiments
    FIXED_SETTINGS = {
        'fairness_weight': 2.0,
        'starvation_weight': 0.5,  # Validated from Exp 009
        'utility_weight': 1.0,
        'soft_threshold': 0.0,  # DISABLED (Exp 011 finding)
        'normalize_scores': True,
        'gamma': 0.5,
        'enable_diagnostics': False
    }
    
    # Worker counts to test
    worker_counts = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000]
    FIXED_TASK_COUNT = 20000
    
    print(f"[CONFIGURATION]")
    print(f"   λ₁ (Fairness):    {FIXED_SETTINGS['fairness_weight']}")
    print(f"   λ₂ (Starvation):  {FIXED_SETTINGS['starvation_weight']}")
    print(f"   λ₃ (Utility):     {FIXED_SETTINGS['utility_weight']}")
    print(f"   θ (Threshold):    {FIXED_SETTINGS['soft_threshold']} (DISABLED)")
    print(f"   Normalize:        {FIXED_SETTINGS['normalize_scores']}")
    print(f"   γ (EWMA):         {FIXED_SETTINGS['gamma']}")
    print()
    
    # Display experiment matrix
    print("Experiment Matrix:")
    print("-" * 80)
    print(f"{'ID':<4} {'Workers':<10} {'Tasks':<10} {'Tasks/Worker':<15} {'Expected TAR':<15}")
    print("-" * 80)
    for i, worker_count in enumerate(worker_counts, 1):
        tasks_per_worker = FIXED_TASK_COUNT / worker_count
        print(f"{i:<4} {worker_count:<10,} {FIXED_TASK_COUNT:<10,} {tasks_per_worker:<15.1f} {'>90%':<15}")
    print("-" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Load full 3-hour peak dataset
    # ========================================================================
    print("[STEP 1] Loading 3-hour peak dataset...")
    data_path = project_root / "data" / "didi"
    all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
    print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")
    print()
    
    # ========================================================================
    # STEP 2: Stratified temporal sampling
    # ========================================================================
    print("[STEP 2] Performing stratified temporal sampling...")
    print()
    
    sampled_tasks, worker_samples = stratified_temporal_sample(
        all_workers=all_workers,
        all_tasks=all_tasks,
        target_tasks=FIXED_TASK_COUNT,
        worker_counts=worker_counts,
        num_bins=12,
        seed=42
    )
    
    print(f"✅ Sampling complete!")
    print(f"   Tasks: {len(sampled_tasks):,} (FIXED for all experiments)")
    print(f"   Worker samples: {len(worker_samples)} (one per worker count)")
    print()
    
    # ========================================================================
    # STEP 3: Run experiments
    # ========================================================================
    print("[STEP 3] Running experiments...")
    print()
    
    results = []
    successful_runs = 0
    
    for i, worker_count in enumerate(worker_counts, 1):
        exp_start = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(worker_counts)}: {worker_count:,} workers")
        print(f"{'='*80}")
        print(f"Tasks: {len(sampled_tasks):,}")
        print(f"Tasks per worker: {len(sampled_tasks) / worker_count:.1f}")
        print()
        
        try:
            # Get stratified worker sample (deep copy to avoid mutation)
            workers = copy.deepcopy(worker_samples[worker_count])
            
            # CRITICAL: Deep copy tasks to avoid mutation across experiments
            tasks = copy.deepcopy(sampled_tasks)
            
            print(f"Workers sampled: {len(workers):,}")
            print(f"Tasks prepared: {len(tasks):,} (fresh copy)")
            print(f"Running simulation...")
            
            # Update strategy params
            STRATEGY_PARAMS['composite'].update(FIXED_SETTINGS)
            
            # Run simulation
            cfg = get_simulation_config()
            summary = run_simulation(workers, tasks, sim_config=cfg)
            
            exp_end = datetime.now()
            exp_duration = (exp_end - exp_start).total_seconds()
            
            # Calculate metrics
            completed = summary.get('completed_tasks', 0)
            tar = completed / len(tasks)
            jfi = summary.get('final_jains_fairness_index', 0)
            
            print(f"\n✅ Experiment {i} complete in {exp_duration/60:.1f} minutes")
            print(f"   Completed: {completed:,} / {len(tasks):,} tasks")
            print(f"   TAR: {tar:.1%}, JFI: {jfi:.3f}, Gini: {summary.get('tasks_per_worker_gini', 0):.3f}")
            print(f"   Mean Wait: {summary.get('avg_wait_time_minutes', 0):.1f} min, Utilization: {summary.get('mean_worker_utilization', 0):.1%}")
            
            # Store results with all v2.0 metrics
            result = {
                'experiment_id': i,
                'worker_count': worker_count,
                'task_count': len(tasks),
                'tasks_per_worker_ratio': len(tasks) / worker_count,
                
                # Configuration
                'fairness_weight': FIXED_SETTINGS['fairness_weight'],
                'starvation_weight': FIXED_SETTINGS['starvation_weight'],
                'utility_weight': FIXED_SETTINGS['utility_weight'],
                'soft_threshold': FIXED_SETTINGS['soft_threshold'],
                'normalize_scores': FIXED_SETTINGS['normalize_scores'],
                'gamma': FIXED_SETTINGS['gamma'],
                
                # Core metrics
                'completed_tasks': completed,
                'task_assignment_ratio': tar,
                'jains_fairness_index': jfi,
                
                # Task wait time distribution (Tier 1)
                'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
                'std_task_wait_time_min': summary.get('std_wait_time_minutes', 0),
                'p90_task_wait_time_min': summary.get('p90_wait_time_minutes', 0),
                'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
                'max_task_wait_time_min': summary.get('max_wait_time_minutes', 0),
                'cv_task_wait_time': summary.get('cv_wait_time', 0),
                
                # Worker idle time distribution (Tier 1)
                'mean_worker_idle_time_min': summary.get('mean_worker_idle_time_min', 0),
                'std_worker_idle_time_min': summary.get('std_worker_idle_time_min', 0),
                'p90_worker_idle_time_min': summary.get('p90_worker_idle_time_min', 0),
                'max_worker_idle_time_min': summary.get('max_worker_idle_time_min', 0),
                'cv_worker_idle_time': summary.get('cv_worker_idle_time', 0),
                
                # Worker task distribution (Tier 2)
                'tasks_per_worker_mean': summary.get('tasks_per_worker_mean', 0),
                'tasks_per_worker_std': summary.get('tasks_per_worker_std', 0),
                'tasks_per_worker_cv': summary.get('tasks_per_worker_cv', 0),
                'tasks_per_worker_gini': summary.get('tasks_per_worker_gini', 0),
                'tasks_per_worker_p10': summary.get('tasks_per_worker_p10', 0),
                'tasks_per_worker_p50': summary.get('tasks_per_worker_p50', 0),
                'tasks_per_worker_p90': summary.get('tasks_per_worker_p90', 0),
                'pct_workers_zero_tasks': summary.get('pct_workers_zero_tasks', 0),
                'pct_workers_single_task': summary.get('pct_workers_single_task', 0),
                
                # Worker utilization (Tier 2)
                'mean_worker_utilization': summary.get('mean_worker_utilization', 0),
                'std_worker_utilization': summary.get('std_worker_utilization', 0),
                'p10_worker_utilization': summary.get('p10_worker_utilization', 0),
                'p90_worker_utilization': summary.get('p90_worker_utilization', 0),
                
                # Pickup distance distribution (Tier 2)
                'mean_pickup_distance_km': summary.get('avg_pickup_distance_km', 0),
                'std_pickup_distance_km': summary.get('std_pickup_distance_km', 0),
                'p90_pickup_distance_km': summary.get('p90_pickup_distance_km', 0),
                'max_pickup_distance_km': summary.get('max_pickup_distance_km', 0),
                
                # Task deferrals (Tier 2)
                'total_deferrals': summary.get('total_deferrals', 0),
                'pct_tasks_deferred': summary.get('pct_tasks_deferred', 0),
                'mean_deferrals_per_task': summary.get('mean_deferrals_per_task', 0),
                'max_deferrals_per_task': summary.get('max_deferrals_per_task', 0),
                
                # Assignment timing (Tier 2)
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
            result_file = output_dir / f"exp_{i:03d}_{worker_count}workers_summary.json"
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
            remaining_exps = len(worker_counts) - i
            eta_seconds = remaining_exps * avg_time_per_exp
            eta = datetime.now() + pd.Timedelta(seconds=eta_seconds)
            
            print(f"\n📊 Progress: {i}/{len(worker_counts)} ({i/len(worker_counts)*100:.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f} min")
            print(f"   Avg per exp: {avg_time_per_exp/60:.1f} min")
            print(f"   ETA: {eta.strftime('%H:%M:%S')} (~{eta_seconds/60:.0f} min remaining)")
            
        except Exception as e:
            print(f"\n❌ Experiment {i} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========================================================================
    # STEP 4: Save aggregate results
    # ========================================================================
    if results:
        df = pd.DataFrame(results)
        csv_path = output_dir.parent / "experiment_012_aggregate_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Aggregate results saved to {csv_path}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 012 COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {successful_runs}/{len(worker_counts)}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average per experiment: {total_duration/len(worker_counts)/60:.1f} minutes")
    print(f"\nOutput directory: {output_dir}")
    print(f"Aggregate CSV: {csv_path if results else 'N/A'}")
    
    # Quick summary of key findings
    if results and len(results) >= 2:
        print("\n" + "=" * 80)
        print("QUICK INSIGHTS")
        print("=" * 80)
        
        df = pd.DataFrame(results)
        
        # Find optimal worker count (>90% TAR with highest tasks/worker)
        high_tar = df[df['task_assignment_ratio'] >= 0.90]
        if not high_tar.empty:
            optimal = high_tar.loc[high_tar['tasks_per_worker_ratio'].idxmax()]
            print(f"\n🎯 Optimal Worker Count: {optimal['worker_count']:,.0f}")
            print(f"   TAR: {optimal['task_assignment_ratio']:.1%}")
            print(f"   Tasks/Worker: {optimal['tasks_per_worker_ratio']:.1f}")
            print(f"   Gini: {optimal['tasks_per_worker_gini']:.3f}")
            print(f"   Mean Wait: {optimal['mean_task_wait_time_min']:.1f} min")
            print(f"   Utilization: {optimal['mean_worker_utilization']:.1%}")
        
        # TAR range
        print(f"\n📊 TAR Range: {df['task_assignment_ratio'].min():.1%} - {df['task_assignment_ratio'].max():.1%}")
        print(f"📊 Gini Range: {df['tasks_per_worker_gini'].min():.3f} - {df['tasks_per_worker_gini'].max():.3f}")
        
        # Utilization trend
        print(f"📊 Utilization: {df['mean_worker_utilization'].max():.1%} (2K) → {df['mean_worker_utilization'].min():.1%} (15K)")
    
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_012()


