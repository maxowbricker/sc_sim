#!/usr/bin/env python3
"""
Experiment 010: Pareto Frontier High-Resolution Sweep
======================================================

This experiment conducts a focused high-resolution sweep of the critical gap
in the fairness-efficiency space identified in Experiment 009: λ₁ between 2.0 and 5.0.

Design Philosophy:
- Focus on the "knee of the curve" in the Pareto frontier
- High resolution in the critical range (λ₁: 2.5-4.5)
- Test interaction with utility weight (λ₃: 0.5-2.0)
- Fix validated parameters from previous experiments

Experimental Design:
- Total Experiments: 21 (1 Greedy + 20 Composite)
- Grid: 5 λ₁ values × 4 λ₃ values
- Fixed: λ₂=0.5, θ=0.5, normalize_scores=True, gamma=0.5

Expected Duration: ~8-9 hours (21 experiments × ~25 min each)
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


def run_experiment_010():
    """Run Experiment 010: Pareto Frontier High-Resolution Sweep."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("EXPERIMENT 010: Pareto Frontier High-Resolution Sweep")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[OBJECTIVE] High-resolution mapping of critical λ₁ range (2.5-4.5)")
    print("[FOCUS] Find the 'knee' of the fairness-efficiency Pareto curve")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / f"exp_010_{timestamp}"
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
        'normalize_scores': True,      # Essential fix from Experiment 008
        'enable_diagnostics': False,   # Disabled for fast path performance
        'gamma': 0.5,                  # Stable value from Experiment 007
        'starvation_weight': 0.5,      # Validated "safety net" value from Exp 009
        'soft_threshold': 0.5,         # Minimal impact (validated in Exp 009)
        'k': 15                        # Standard nearest neighbor count
    }
    
    print("[FIXED SETTINGS FOR ALL COMPOSITE RUNS]")
    for key, value in FIXED_SETTINGS.items():
        print(f"   {key}: {value}")
    print()
    
    # Define all 21 experimental configurations
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
    # GROUP B: PARETO FRONTIER SWEEP (20 experiments)
    # High-resolution 5 × 4 grid
    # ========================================================================
    print("[GROUP B] Pareto Frontier High-Resolution Sweep")
    print("   Purpose: Map critical fairness-efficiency trade-off region")
    print("   Fixed: L2=0.5, Threshold=0.5")
    print("   Grid: λ₁ ∈ [2.5, 3.0, 3.5, 4.0, 4.5] × λ₃ ∈ [0.5, 1.0, 1.5, 2.0]")
    print()
    
    # Define grid points
    l1_values = [2.5, 3.0, 3.5, 4.0, 4.5]  # Critical gap range
    l3_values = [0.5, 1.0, 1.5, 2.0]       # Representative utility spread
    
    for l1 in l1_values:
        for l3 in l3_values:
            experiments.append({
                'id': experiment_id,
                'group': 'B',
                'name': f'ParetoSweep_L1_{l1}_L3_{l3}',
                'description': f'L1={l1}, L2=0.5, L3={l3}, Threshold=0.5',
                'strategy': 'composite',
                'config_params': {
                    **FIXED_SETTINGS,
                    'fairness_weight': l1,
                    'utility_weight': l3
                }
            })
            experiment_id += 1
    
    print(f"   ✅ {len(l1_values) * len(l3_values)} experiments configured")
    print(f"   Grid resolution: {len(l1_values)} λ₁ values × {len(l3_values)} λ₃ values")
    print()
    
    # ========================================================================
    # EXPERIMENT EXECUTION
    # ========================================================================
    
    print("=" * 80)
    print(f"TOTAL EXPERIMENTS CONFIGURED: {len(experiments)}")
    print("=" * 80)
    print()
    
    # Save experiment manifest
    manifest = {
        'experiment_name': 'Experiment 010: Pareto Frontier High-Resolution Sweep',
        'timestamp': timestamp,
        'total_experiments': len(experiments),
        'fixed_settings': FIXED_SETTINGS,
        'dataset': 'didi',
        'max_workers': 15000,
        'max_tasks': 20000,
        'grid_l1': l1_values,
        'grid_l3': l3_values,
        'focus': 'High-resolution mapping of critical λ₁ range (2.5-4.5)',
        'predecessor': 'Experiment 009 (42 experiments)'
    }
    
    manifest_file = output_dir / 'experiment_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Manifest saved: {manifest_file}")
    print()
    
    # Run experiments
    results = []
    successful_runs = 0
    failed_runs = 0
    
    for exp_config in experiments:
        exp_id = exp_config['id']
        exp_name = exp_config['name']
        exp_group = exp_config['group']
        
        print(f"\n{'='*80}")
        print(f"[EXPERIMENT {exp_id}/{len(experiments)}] {exp_name} (Group {exp_group})")
        print(f"{'='*80}")
        print(f"Description: {exp_config['description']}")
        
        exp_start_time = time.time()
        
        try:
            # Create configuration
            if exp_config['strategy'] == 'greedy':
                config = create_composite_config(
                    assignment_strategy='greedy'
                )
            else:
                config = create_composite_config(
                    assignment_strategy='composite',
                    **exp_config['config_params']
                )
            
            # Run simulation
            print(f"\n🚀 Starting simulation...")
            sim = Simulation(config, workers_df.copy(), tasks_df.copy())
            summary = sim.run()
            
            exp_duration = time.time() - exp_start_time
            
            print(f"\n✅ Simulation complete in {exp_duration:.1f} seconds")
            print(f"   Completed tasks: {summary['completed_tasks']}/{summary['total_tasks']}")
            print(f"   JFI: {summary['jfi']:.4f}")
            print(f"   Mean wait time: {summary['avg_wait_time_minutes']:.2f} min")
            print(f"   TAR: {summary['task_assignment_ratio']*100:.2f}%")
            
            # Store results (using .get() with defaults for safety)
            result = {
                'experiment_id': exp_id,
                'group': exp_group,
                'name': exp_name,
                'description': exp_config['description'],
                'strategy': exp_config['strategy'],
                
                # Experimental parameters
                'fairness_weight': exp_config['config_params'].get('fairness_weight'),
                'starvation_weight': exp_config['config_params'].get('starvation_weight'),
                'utility_weight': exp_config['config_params'].get('utility_weight'),
                'soft_threshold': exp_config['config_params'].get('soft_threshold'),
                'normalize_scores': exp_config['config_params'].get('normalize_scores'),
                'gamma': exp_config['config_params'].get('gamma'),
                
                # Primary metrics (using .get() to avoid KeyError)
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
                
                # Additional metrics
                'empty_km_ratio': summary.get('empty_km_share', 0),
                'ewma_cv': summary.get('ewma_cv', 0),
                'max_wait_time': summary.get('p90_wait_time_minutes', 0),
                
                # Metadata
                'duration_seconds': exp_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Save individual result with full summary (using default=str for safety)
            result_file = output_dir / f"exp_{exp_id:03d}_{exp_name}_summary.json"
            with open(result_file, 'w') as f:
                full_result = {
                    **result,
                    'full_summary': {k: v for k, v in summary.items() if k not in ['metric_tracker', 'diagnostic_tracker', 'workers_df']}
                }
                json.dump(full_result, f, indent=2, default=str)
            
            # Save worker-level data if available
            if 'workers_df' in summary:
                worker_file = output_dir / f"exp_{exp_id:03d}_{exp_name}_workers.csv"
                summary['workers_df'].to_csv(worker_file, index=False)
            
            successful_runs += 1
            
        except Exception as e:
            failed_runs += 1
            exp_duration = time.time() - exp_start_time
            
            print(f"\n❌ FAILED after {exp_duration:.1f} seconds")
            print(f"   Error: {str(e)}")
            print(f"   Traceback:")
            traceback.print_exc()
            
            # Store failure record
            result = {
                'experiment_id': exp_id,
                'group': exp_group,
                'name': exp_name,
                'status': 'FAILED',
                'error': str(e),
                'duration_seconds': exp_duration
            }
            results.append(result)
        
        # Progress update
        elapsed_time = time.time() - start_time.timestamp()
        avg_time_per_exp = elapsed_time / exp_id
        remaining_exps = len(experiments) - exp_id
        estimated_remaining = avg_time_per_exp * remaining_exps
        
        print(f"\n[PROGRESS] {exp_id}/{len(experiments)} complete ({successful_runs} successful, {failed_runs} failed)")
        print(f"[TIME] Elapsed: {elapsed_time/3600:.1f}h | Estimated remaining: {estimated_remaining/3600:.1f}h")
    
    # Save aggregate results
    print(f"\n{'='*80}")
    print("SAVING AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame([r for r in results if 'jains_fairness_index' in r])
    aggregate_file = output_dir / 'experiment_010_aggregate_results.csv'
    results_df.to_csv(aggregate_file, index=False)
    print(f"✅ Saved: {aggregate_file}")
    
    # Final summary
    total_duration = time.time() - start_time.timestamp()
    end_time = datetime.now()
    
    print(f"\n{'='*80}")
    print("EXPERIMENT 010 COMPLETE")
    print(f"{'='*80}")
    print(f"[END TIME] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION] {total_duration/3600:.2f} hours")
    print(f"[SUCCESS] {successful_runs}/{len(experiments)} experiments")
    print(f"[FAILED] {failed_runs}/{len(experiments)} experiments")
    print(f"\n[OUTPUT] {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_experiment_010()
