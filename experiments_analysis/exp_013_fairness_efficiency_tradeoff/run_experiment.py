#!/usr/bin/env python3
"""
Experiment 013: High-Resolution Fairness-Efficiency Trade-off Mapping
======================================================================

Maps the Pareto frontier at high resolution using a 10×7 grid of λ₁ × λ₃ values.

Key Innovation: High-resolution sweep focused on promising parameter region
identified in Experiments 009 and 012.

Parameter Grid:
- λ₁ (Fairness): [2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]
- λ₃ (Utility): [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

Configuration: 4K workers, 20K tasks (validated optimal from Exp 012)
Total: 73 experiments (~8 hours)
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


def run_experiment_013():
    """Run Experiment 013: High-Resolution Fairness-Efficiency Trade-off Mapping."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("EXPERIMENT 013: High-Resolution Fairness-Efficiency Trade-off Mapping")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[OBJECTIVE] Map Pareto frontier at high resolution (λ₁ × λ₃ grid)")
    print("[METHOD] Stratified temporal sampling with validated 4K workers")
    print("[CONFIGURATION] λ₂=0.5, θ=0.0, normalize_scores=True")
    print("[GRID] 10 × 7 = 70 configs + 1 greedy + 2 balance points")
    print("[TOTAL EXPERIMENTS] 73")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / f"exp_013_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {output_dir}")
    print()
    
    # Fixed settings for all composite experiments
    FIXED_SETTINGS = {
        'starvation_weight': 0.5,  # Validated from Exp 009 & 012
        'soft_threshold': 0.0,  # DISABLED (Exp 011 & 012 finding)
        'normalize_scores': True,
        'gamma': 0.5,
        'enable_diagnostics': False
    }
    
    # Worker/task configuration (validated from Exp 012)
    WORKER_COUNT = 4000
    TASK_COUNT = 20000
    
    print(f"[FIXED CONFIGURATION]")
    print(f"   Workers:          {WORKER_COUNT:,}")
    print(f"   Tasks:            {TASK_COUNT:,}")
    print(f"   Tasks/Worker:     {TASK_COUNT / WORKER_COUNT:.1f}")
    print(f"   λ₂ (Starvation):  {FIXED_SETTINGS['starvation_weight']}")
    print(f"   θ (Threshold):    {FIXED_SETTINGS['soft_threshold']} (DISABLED)")
    print(f"   Normalize:        {FIXED_SETTINGS['normalize_scores']}")
    print(f"   γ (EWMA):         {FIXED_SETTINGS['gamma']}")
    print()
    
    # ========================================================================
    # Define experiment configurations
    # ========================================================================
    
    experiments = []
    exp_id = 1
    
    # ========================================================================
    # GROUP A: Greedy Baseline (1 experiment)
    # ========================================================================
    print("[GROUP A] Greedy Baseline")
    experiments.append({
        'id': exp_id,
        'group': 'A',
        'name': 'Greedy_Baseline',
        'description': 'Greedy baseline for efficiency reference',
        'strategy': 'greedy',
        'config_params': {}
    })
    exp_id += 1
    print(f"   ✅ 1 experiment configured")
    print()
    
    # ========================================================================
    # GROUP B: High-Resolution λ₁ × λ₃ Grid (70 experiments)
    # ========================================================================
    print("[GROUP B] High-Resolution λ₁ × λ₃ Grid")
    
    lambda1_values = [2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]
    lambda3_values = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    print(f"   λ₁ values: {lambda1_values}")
    print(f"   λ₃ values: {lambda3_values}")
    print(f"   Grid size: {len(lambda1_values)} × {len(lambda3_values)} = {len(lambda1_values) * len(lambda3_values)}")
    print()
    
    for l1 in lambda1_values:
        for l3 in lambda3_values:
            experiments.append({
                'id': exp_id,
                'group': 'B',
                'name': f'L1_{l1}_L3_{l3}',
                'description': f'λ₁={l1}, λ₂=0.5, λ₃={l3}',
                'strategy': 'composite',
                'config_params': {
                    **FIXED_SETTINGS,
                    'fairness_weight': l1,
                    'utility_weight': l3
                }
            })
            exp_id += 1
    
    print(f"   ✅ {len(lambda1_values) * len(lambda3_values)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP C: Balance Point Runs (2 experiments)
    # ========================================================================
    print("[GROUP C] Balance Point Runs (Equal/Near-Equal Weights)")
    
    balance_points = [
        (2.5, 2.5, "Equal weights (moderate)"),
        (2.25, 2.75, "Near-equal (efficiency-biased)")
    ]
    
    for l1, l3, desc in balance_points:
        experiments.append({
            'id': exp_id,
            'group': 'C',
            'name': f'Balance_L1_{l1}_L3_{l3}',
            'description': f'λ₁={l1}, λ₂=0.5, λ₃={l3} - {desc}',
            'strategy': 'composite',
            'config_params': {
                **FIXED_SETTINGS,
                'fairness_weight': l1,
                'utility_weight': l3
            }
        })
        exp_id += 1
    
    print(f"   ✅ {len(balance_points)} experiments configured")
    print()
    
    # ========================================================================
    # Display experiment matrix summary
    # ========================================================================
    print("=" * 80)
    print(f"[TOTAL EXPERIMENTS] {len(experiments)}")
    print("=" * 80)
    print()
    
    # Group summary
    group_counts = {}
    for exp in experiments:
        group = exp['group']
        group_counts[group] = group_counts.get(group, 0) + 1
    
    print("[GROUP SUMMARY]")
    for group in sorted(group_counts.keys()):
        print(f"   Group {group}: {group_counts[group]} experiments")
    print()
    
    # Estimate duration
    avg_time_per_exp = 6.6  # minutes (based on Exp 012 with 4K workers)
    estimated_total_hours = (len(experiments) * avg_time_per_exp) / 60
    print(f"[ESTIMATED DURATION] {estimated_total_hours:.1f} hours")
    print(f"   (Assuming ~{avg_time_per_exp} min per experiment)")
    print()
    print("[STARTING EXPERIMENTS]")
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
        target_tasks=TASK_COUNT,
        worker_counts=[WORKER_COUNT],
        num_bins=12,
        seed=42
    )
    
    # Get the 4K worker sample
    sampled_workers = worker_samples[WORKER_COUNT]
    
    print(f"✅ Sampling complete!")
    print(f"   Tasks: {len(sampled_tasks):,} (FIXED for all experiments)")
    print(f"   Workers: {len(sampled_workers):,} (FIXED for all experiments)")
    print()
    
    # ========================================================================
    # STEP 3: Run experiments
    # ========================================================================
    print("[STEP 3] Running experiments...")
    print()
    
    results = []
    successful_runs = 0
    
    for i, exp in enumerate(experiments, 1):
        exp_start = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")
        print(f"[ID] {exp['id']}")
        print(f"[GROUP] {exp['group']}")
        print(f"[STRATEGY] {exp['strategy']}")
        print(f"[DESCRIPTION] {exp['description']}")
        
        if exp['strategy'] == 'composite':
            params = exp['config_params']
            print(f"[PARAMETERS]")
            print(f"   λ₁ (Fairness):     {params.get('fairness_weight', 'N/A')}")
            print(f"   λ₂ (Starvation):   {params.get('starvation_weight', 'N/A')}")
            print(f"   λ₃ (Utility):      {params.get('utility_weight', 'N/A')}")
            print(f"   θ (Threshold):     {params.get('soft_threshold', 'N/A')}")
            print(f"   Normalize Scores:  {params.get('normalize_scores', 'N/A')}")
        
        print()
        
        try:
            # CRITICAL: Deep copy to avoid mutation across experiments
            workers = copy.deepcopy(sampled_workers)
            tasks = copy.deepcopy(sampled_tasks)
            
            print(f"Workers: {len(workers):,}")
            print(f"Tasks: {len(tasks):,} (fresh copy)")
            print(f"Running simulation...")
            
            # Update strategy params if composite
            if exp['strategy'] == 'composite':
                STRATEGY_PARAMS['composite'].update(exp['config_params'])
            
            # Run simulation
            cfg = get_simulation_config()
            if exp['strategy'] == 'greedy':
                cfg['assignment_strategy'] = 'greedy'
            
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
                'experiment_id': exp['id'],
                'group': exp['group'],
                'name': exp['name'],
                'description': exp['description'],
                'strategy': exp['strategy'],
                
                # Configuration
                'worker_count': WORKER_COUNT,
                'task_count': len(tasks),
                'tasks_per_worker_ratio': len(tasks) / WORKER_COUNT,
                'fairness_weight': exp['config_params'].get('fairness_weight') if exp['strategy'] == 'composite' else None,
                'starvation_weight': exp['config_params'].get('starvation_weight') if exp['strategy'] == 'composite' else None,
                'utility_weight': exp['config_params'].get('utility_weight') if exp['strategy'] == 'composite' else None,
                'soft_threshold': exp['config_params'].get('soft_threshold') if exp['strategy'] == 'composite' else None,
                'normalize_scores': exp['config_params'].get('normalize_scores') if exp['strategy'] == 'composite' else None,
                'gamma': exp['config_params'].get('gamma') if exp['strategy'] == 'composite' else None,
                
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
            
            # Save individual experiment results
            exp_filename = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_summary.json"
            with open(exp_filename, 'w') as f:
                full_result = {
                    **result,
                    'full_summary': {k: v for k, v in summary.items() 
                                   if k not in ['metric_tracker', 'diagnostic_tracker', 'workers_df']}
                }
                json.dump(full_result, f, indent=2, default=str)
            
            print(f"   💾 Saved to {exp_filename.name}")
            
            # Progress update
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time_per_exp_actual = elapsed / i
            remaining_exps = len(experiments) - i
            eta_seconds = remaining_exps * avg_time_per_exp_actual
            eta = datetime.now() + pd.Timedelta(seconds=eta_seconds)
            
            print(f"\n📊 Progress: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f} min")
            print(f"   Avg per exp: {avg_time_per_exp_actual/60:.1f} min")
            print(f"   ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} (~{eta_seconds/60:.0f} min remaining)")
            
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
        csv_path = output_dir.parent / "experiment_013_aggregate_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Aggregate results saved to {csv_path}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 013 COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {successful_runs}/{len(experiments)}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average per experiment: {total_duration/len(experiments)/60:.1f} minutes")
    print()
    print(f"Output directory: {output_dir}")
    print(f"Aggregate CSV: {csv_path}")
    print("=" * 80)


if __name__ == '__main__':
    run_experiment_013()



