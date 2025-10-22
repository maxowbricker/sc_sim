#!/usr/bin/env python3
"""
Experiment 009: Comprehensive Parameter Sweep (Post-Normalization)
==================================================================

This experiment conducts a comprehensive parameter sweep with the normalized scoring
fix from Experiment 008. With score normalization enabled, we can now reliably explore
the fairness-efficiency trade-off space.

Experimental Design:
- Total Experiments: 42 (1 Greedy + 41 Composite configurations)
- Fixed Settings: normalize_scores=True, gamma=0.5, enable_diagnostics=False
- Single run per configuration (no replications) for maximum parameter space coverage

Expected Duration: ~6-8 hours (42 experiments × ~10 min each)
"""

import json
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data


def run_experiment_009():
    """Run Experiment 009: Comprehensive Parameter Sweep (Post-Normalization)."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("EXPERIMENT 009: Comprehensive Parameter Sweep (Post-Normalization)")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[OBJECTIVE] Comprehensive fairness-efficiency parameter space exploration")
    print("[INNOVATION] Normalized scoring (fixes idle time paradox from Exp 006)")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / f"exp_009_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT DIR] {output_dir}")
    print()
    
    # Load dataset once for all experiments
    print("[LOADING DATASET]")
    print("   Dataset: DiDi (15,000 workers, 20,000 tasks)")
    workers_df, tasks_df = load_data('didi', max_workers=15000, max_tasks=20000)
    print(f"   ✅ Loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print()
    
    # Fixed global settings for all composite runs
    FIXED_SETTINGS = {
        'normalize_scores': True,      # Key fix from Experiment 008
        'enable_diagnostics': False,   # Disabled for fast path performance
        'gamma': 0.5,                  # Stable value from Experiment 007
        'k': 15                        # Standard nearest neighbor count
    }
    
    print("[FIXED SETTINGS FOR ALL COMPOSITE RUNS]")
    for key, value in FIXED_SETTINGS.items():
        print(f"   {key}: {value}")
    print()
    
    # Define all 42 experimental configurations
    experiments = []
    experiment_id = 1
    
    # ========================================================================
    # GROUP A: GREEDY BASELINE (1 experiment)
    # ========================================================================
    print("[GROUP A] Greedy Baseline")
    print("   Purpose: Efficiency reference")
    experiments.append({
        'id': experiment_id,
        'group': 'A',
        'name': 'Greedy_Baseline',
        'description': 'Greedy baseline for efficiency comparison',
        'strategy': 'greedy',
        'config_params': {}
    })
    experiment_id += 1
    print(f"   ✅ 1 experiment configured")
    print()
    
    # ========================================================================
    # GROUP B: L1 × L3 GRID SWEEP (12 experiments)
    # Core RQ1 Mapping: Fairness (L1) vs Utility (L3)
    # ========================================================================
    print("[GROUP B] L1 × L3 Grid Sweep (Core RQ1 Mapping)")
    print("   Purpose: Map fairness-efficiency trade-off space")
    print("   Fixed: L2=0.8, Threshold=0.5")
    
    l1_values = [0.0, 0.5, 1.0, 2.0]
    l3_values = [0.5, 1.0, 2.0]
    
    for l1 in l1_values:
        for l3 in l3_values:
            experiments.append({
                'id': experiment_id,
                'group': 'B',
                'name': f'L1_{l1}_L3_{l3}',
                'description': f'L1={l1}, L2=0.8, L3={l3}, Threshold=0.5',
                'strategy': 'composite',
                'config_params': {
                    **FIXED_SETTINGS,
                    'fairness_weight': l1,
                    'starvation_weight': 0.8,
                    'utility_weight': l3,
                    'soft_threshold': 0.5
                }
            })
            experiment_id += 1
    
    print(f"   ✅ {len(l1_values) * len(l3_values)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP C: STARVATION (L2) ABLATION (4 experiments)
    # Core RQ3: Does starvation weight matter?
    # ========================================================================
    print("[GROUP C] Starvation L2 Ablation (Core RQ3)")
    print("   Purpose: Test impact of starvation weight")
    print("   Fixed: L1=1.0, L3=1.0, Threshold=0.5")
    
    l2_values = [0.0, 0.5, 1.0, 2.0]
    
    for l2 in l2_values:
        experiments.append({
            'id': experiment_id,
            'group': 'C',
            'name': f'L2_Ablation_{l2}',
            'description': f'L1=1.0, L2={l2}, L3=1.0, Threshold=0.5',
            'strategy': 'composite',
            'config_params': {
                **FIXED_SETTINGS,
                'fairness_weight': 1.0,
                'starvation_weight': l2,
                'utility_weight': 1.0,
                'soft_threshold': 0.5
            }
        })
        experiment_id += 1
    
    print(f"   ✅ {len(l2_values)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP D: SOFT THRESHOLD SWEEP (4 experiments)
    # Core RQ3: Threshold sensitivity analysis
    # ========================================================================
    print("[GROUP D] Soft Threshold Sweep (Core RQ3)")
    print("   Purpose: Test threshold sensitivity")
    print("   Fixed: L1=1.0, L2=0.8, L3=1.0")
    
    threshold_values = [0.1, 0.3, 0.6, 0.9]
    
    for threshold in threshold_values:
        experiments.append({
            'id': experiment_id,
            'group': 'D',
            'name': f'Threshold_{threshold}',
            'description': f'L1=1.0, L2=0.8, L3=1.0, Threshold={threshold}',
            'strategy': 'composite',
            'config_params': {
                **FIXED_SETTINGS,
                'fairness_weight': 1.0,
                'starvation_weight': 0.8,
                'utility_weight': 1.0,
                'soft_threshold': threshold
            }
        })
        experiment_id += 1
    
    print(f"   ✅ {len(threshold_values)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP E: BALANCED GRID SWEEP (9 experiments)
    # Fine-grained exploration around balanced region
    # ========================================================================
    print("[GROUP E] Balanced Grid Sweep")
    print("   Purpose: Fine-grained exploration near balanced region")
    print("   Fixed: L2=0.8, Threshold=0.5")
    
    l1_balanced = [0.2, 0.4, 0.6]
    l3_balanced = [1.2, 1.6, 2.0]
    
    for l1 in l1_balanced:
        for l3 in l3_balanced:
            experiments.append({
                'id': experiment_id,
                'group': 'E',
                'name': f'Balanced_L1_{l1}_L3_{l3}',
                'description': f'L1={l1}, L2=0.8, L3={l3}, Threshold=0.5',
                'strategy': 'composite',
                'config_params': {
                    **FIXED_SETTINGS,
                    'fairness_weight': l1,
                    'starvation_weight': 0.8,
                    'utility_weight': l3,
                    'soft_threshold': 0.5
                }
            })
            experiment_id += 1
    
    print(f"   ✅ {len(l1_balanced) * len(l3_balanced)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP F: HIGH-FAIRNESS EDGE (4 experiments)
    # Extreme fairness-focused configurations (L1=5.0)
    # ========================================================================
    print("[GROUP F] High-Fairness Edge (L1=5.0)")
    print("   Purpose: Test extreme fairness-focused configurations")
    print("   Fixed: L2=0.8, Threshold=0.5")
    
    l3_high_fairness = [0.5, 1.0, 1.5, 2.0]
    
    for l3 in l3_high_fairness:
        experiments.append({
            'id': experiment_id,
            'group': 'F',
            'name': f'HighFairness_L3_{l3}',
            'description': f'L1=5.0, L2=0.8, L3={l3}, Threshold=0.5',
            'strategy': 'composite',
            'config_params': {
                **FIXED_SETTINGS,
                'fairness_weight': 5.0,
                'starvation_weight': 0.8,
                'utility_weight': l3,
                'soft_threshold': 0.5
            }
        })
        experiment_id += 1
    
    print(f"   ✅ {len(l3_high_fairness)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP G: LOW-FAIRNESS EDGE (4 experiments)
    # Near-greedy configurations (L1=0.1)
    # ========================================================================
    print("[GROUP G] Low-Fairness Edge (L1=0.1)")
    print("   Purpose: Test near-greedy efficiency-focused configurations")
    print("   Fixed: L2=0.8, Threshold=0.5")
    
    l3_low_fairness = [0.5, 1.0, 1.5, 2.0]
    
    for l3 in l3_low_fairness:
        experiments.append({
            'id': experiment_id,
            'group': 'G',
            'name': f'LowFairness_L3_{l3}',
            'description': f'L1=0.1, L2=0.8, L3={l3}, Threshold=0.5',
            'strategy': 'composite',
            'config_params': {
                **FIXED_SETTINGS,
                'fairness_weight': 0.1,
                'starvation_weight': 0.8,
                'utility_weight': l3,
                'soft_threshold': 0.5
            }
        })
        experiment_id += 1
    
    print(f"   ✅ {len(l3_low_fairness)} experiments configured")
    print()
    
    # ========================================================================
    # GROUP H: LOW-UTILITY EDGE (4 experiments)
    # Fairness-dominated configurations (L3=0.1)
    # ========================================================================
    print("[GROUP H] Low-Utility Edge (L3=0.1)")
    print("   Purpose: Test fairness-dominated configurations")
    print("   Fixed: L1=1.0, Threshold=0.5")
    
    l2_low_utility = [0.5, 1.0, 1.5, 2.0]
    
    for l2 in l2_low_utility:
        experiments.append({
            'id': experiment_id,
            'group': 'H',
            'name': f'LowUtility_L2_{l2}',
            'description': f'L1=1.0, L2={l2}, L3=0.1, Threshold=0.5',
            'strategy': 'composite',
            'config_params': {
                **FIXED_SETTINGS,
                'fairness_weight': 1.0,
                'starvation_weight': l2,
                'utility_weight': 0.1,
                'soft_threshold': 0.5
            }
        })
        experiment_id += 1
    
    print(f"   ✅ {len(l2_low_utility)} experiments configured")
    print()
    
    # ========================================================================
    # EXPERIMENT EXECUTION
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
    avg_time_per_exp = 10  # minutes (based on Exp 008 performance)
    estimated_total_hours = (len(experiments) * avg_time_per_exp) / 60
    print(f"[ESTIMATED DURATION] {estimated_total_hours:.1f} hours")
    print(f"   (Assuming ~{avg_time_per_exp} min per experiment)")
    print()
    print("[STARTING EXPERIMENTS]")
    print()
    
    # Run experiments
    results = []
    successful_count = 0
    failed_experiments = []
    
    for i, exp in enumerate(experiments, 1):
        exp_start_time = time.time()
        
        print("=" * 80)
        print(f"[EXPERIMENT {i}/{len(experiments)}] {exp['name']}")
        print("=" * 80)
        print(f"[ID] {exp['id']}")
        print(f"[GROUP] {exp['group']}")
        print(f"[STRATEGY] {exp['strategy']}")
        print(f"[DESCRIPTION] {exp['description']}")
        
        if exp['strategy'] == 'composite':
            params = exp['config_params']
            print(f"[PARAMETERS]")
            print(f"   L1 (Fairness):     {params.get('fairness_weight', 'N/A')}")
            print(f"   L2 (Starvation):   {params.get('starvation_weight', 'N/A')}")
            print(f"   L3 (Utility):      {params.get('utility_weight', 'N/A')}")
            print(f"   Soft Threshold:    {params.get('soft_threshold', 'N/A')}")
            print(f"   Normalize Scores:  {params.get('normalize_scores', 'N/A')}")
            print(f"   Gamma (EWMA):      {params.get('gamma', 'N/A')}")
        
        print()
        
        try:
            # Create configuration
            if exp['strategy'] == 'greedy':
                config = create_composite_config(assignment_strategy='greedy')
            else:
                config = create_composite_config(**exp['config_params'])
            
            # Run simulation
            print("[RUNNING] Starting simulation...")
            sim = Simulation(config, workers_df, tasks_df)
            summary = sim.run()
            
            exp_duration = time.time() - exp_start_time
            print(f"[COMPLETE] Duration: {exp_duration/60:.1f} minutes")
            print()
            
            # Extract key metrics
            result = {
                'experiment_id': exp['id'],
                'group': exp['group'],
                'name': exp['name'],
                'description': exp['description'],
                'strategy': exp['strategy'],
                
                # Experimental parameters
                'fairness_weight': exp['config_params'].get('fairness_weight') if exp['strategy'] == 'composite' else None,
                'starvation_weight': exp['config_params'].get('starvation_weight') if exp['strategy'] == 'composite' else None,
                'utility_weight': exp['config_params'].get('utility_weight') if exp['strategy'] == 'composite' else None,
                'soft_threshold': exp['config_params'].get('soft_threshold') if exp['strategy'] == 'composite' else None,
                'normalize_scores': exp['config_params'].get('normalize_scores') if exp['strategy'] == 'composite' else None,
                'gamma': exp['config_params'].get('gamma') if exp['strategy'] == 'composite' else None,
                
                # Primary metrics
                'completed_tasks': summary.get('completed_tasks', 0),
                'task_assignment_ratio': summary.get('completed_tasks', 0) / 20000,
                'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
                'std_task_wait_time_min': summary.get('std_wait_time_minutes', 0),
                'p90_task_wait_time_min': summary.get('p90_wait_time_minutes', 0),
                'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
                'max_task_wait_time_min': summary.get('max_wait_time_minutes', 0),
                'cv_task_wait_time': summary.get('cv_wait_time', 0),
                'mean_worker_idle_time_min': summary.get('mean_worker_idle_time_min', 0),
                'std_worker_idle_time_min': summary.get('std_worker_idle_time_min', 0),
                'p90_worker_idle_time_min': summary.get('p90_worker_idle_time_min', 0),
                'max_worker_idle_time_min': summary.get('max_worker_idle_time_min', 0),
                'cv_worker_idle_time': summary.get('cv_worker_idle_time', 0),
                
                # Tier 1 & 2 Metrics
                'tasks_per_worker_mean': summary.get('tasks_per_worker_mean', 0),
                'tasks_per_worker_std': summary.get('tasks_per_worker_std', 0),
                'tasks_per_worker_cv': summary.get('tasks_per_worker_cv', 0),
                'tasks_per_worker_gini': summary.get('tasks_per_worker_gini', 0),
                'tasks_per_worker_p10': summary.get('tasks_per_worker_p10', 0),
                'tasks_per_worker_p50': summary.get('tasks_per_worker_p50', 0),
                'tasks_per_worker_p90': summary.get('tasks_per_worker_p90', 0),
                'pct_workers_zero_tasks': summary.get('pct_workers_zero_tasks', 0),
                'pct_workers_single_task': summary.get('pct_workers_single_task', 0),
                'std_pickup_distance_km': summary.get('std_pickup_distance_km', 0),
                'p90_pickup_distance_km': summary.get('p90_pickup_distance_km', 0),
                'max_pickup_distance_km': summary.get('max_pickup_distance_km', 0),
                'mean_worker_utilization': summary.get('mean_worker_utilization', 0),
                'std_worker_utilization': summary.get('std_worker_utilization', 0),
                'p10_worker_utilization': summary.get('p10_worker_utilization', 0),
                'p90_worker_utilization': summary.get('p90_worker_utilization', 0),
                'total_deferrals': summary.get('total_deferrals', 0),
                'pct_tasks_deferred': summary.get('pct_tasks_deferred', 0),
                'mean_deferrals_per_task': summary.get('mean_deferrals_per_task', 0),
                'max_deferrals_per_task': summary.get('max_deferrals_per_task', 0),
                'mean_assignment_delay_sec': summary.get('mean_assignment_delay_sec', 0),
                'std_assignment_delay_sec': summary.get('std_assignment_delay_sec', 0),
                'p90_assignment_delay_sec': summary.get('p90_assignment_delay_sec', 0),
                
                'mean_pickup_distance_km': summary.get('avg_pickup_distance_km', 0),
                'total_travel_km': summary.get('total_travel_distance_km', 0),
                'peak_backlog': summary.get('backlog_peak', 0),
                
                # Metadata
                'duration_seconds': exp_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            successful_count += 1
            
            # Save individual experiment results
            exp_filename = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_summary.json"
            with open(exp_filename, 'w') as f:
                full_result = {
                    **result,
                    'full_summary': {k: v for k, v in summary.items() if k not in ['metric_tracker', 'diagnostic_tracker']}
                }
                json.dump(full_result, f, indent=2, default=str)
            
            # Save worker-level data for idle time analysis
            if 'workers_df' in summary:
                worker_filename = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_workers.csv"
                summary['workers_df'].to_csv(worker_filename, index=False)
            
            # Progress update
            elapsed_hours = (time.time() - start_time.timestamp()) / 3600
            remaining_experiments = len(experiments) - i
            avg_time_so_far = elapsed_hours / i * 60  # minutes per experiment
            estimated_remaining_hours = (remaining_experiments * avg_time_so_far) / 60
            
            print(f"[PROGRESS] {i}/{len(experiments)} complete ({successful_count} successful)")
            print(f"[TIME] Elapsed: {elapsed_hours:.1f}h | Estimated remaining: {estimated_remaining_hours:.1f}h")
            print()
            
        except Exception as e:
            print(f"❌ [ERROR] Experiment {exp['id']} failed: {str(e)}")
            print(traceback.format_exc())
            failed_experiments.append({
                'id': exp['id'],
                'name': exp['name'],
                'error': str(e)
            })
            print()
    
    # ========================================================================
    # SAVE AGGREGATE RESULTS
    # ========================================================================
    print("=" * 80)
    print("[SAVING AGGREGATE RESULTS]")
    print("=" * 80)
    
    # Create aggregate DataFrame
    df = pd.DataFrame(results)
    
    # Calculate mean idle times from worker CSVs
    for idx, row in df.iterrows():
        exp_id = row['experiment_id']
        name = row['name']
        worker_file = output_dir / f'exp_{exp_id:03d}_{name}_workers.csv'
        if worker_file.exists():
            workers_df_result = pd.read_csv(worker_file)
            df.at[idx, 'mean_worker_idle_time_min'] = workers_df_result['T_idle'].mean()
    
    # Save aggregate CSV
    aggregate_file = output_dir / "experiment_009_aggregate_results.csv"
    df.to_csv(aggregate_file, index=False)
    print(f"✅ Saved aggregate results: {aggregate_file}")
    
    # Save experiment manifest
    manifest = {
        'experiment_name': 'Experiment 009: Comprehensive Parameter Sweep (Post-Normalization)',
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration_hours': (time.time() - start_time.timestamp()) / 3600,
        'total_experiments': len(experiments),
        'successful_experiments': successful_count,
        'failed_experiments': len(failed_experiments),
        'fixed_settings': FIXED_SETTINGS,
        'dataset': {
            'workers': len(workers_df),
            'tasks': len(tasks_df)
        },
        'output_directory': str(output_dir)
    }
    
    if failed_experiments:
        manifest['failures'] = failed_experiments
    
    manifest_file = output_dir / "experiment_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"✅ Saved experiment manifest: {manifest_file}")
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600
    
    print("=" * 80)
    print("[EXPERIMENT 009 COMPLETE]")
    print("=" * 80)
    print(f"[START TIME]  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[END TIME]    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION]    {total_duration:.2f} hours")
    print()
    print(f"[RESULTS]     {successful_count}/{len(experiments)} experiments successful")
    if failed_experiments:
        print(f"[FAILURES]    {len(failed_experiments)} experiments failed:")
        for failure in failed_experiments:
            print(f"              - Exp {failure['id']} ({failure['name']}): {failure['error']}")
    print()
    print(f"[OUTPUT DIR]  {output_dir}")
    print()
    print("[NEXT STEPS]")
    print("   1. Review aggregate results CSV")
    print("   2. Run analysis notebook (create with create_analysis_notebook.py)")
    print("   3. Generate visualizations for fairness-efficiency trade-offs")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_009()

