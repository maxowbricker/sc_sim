#!/usr/bin/env python3
"""
Experiment 009 Part 2: Experiments 22-42 (Continuation)
========================================================

Continues from experiment 21, running experiments 22-42.
Uses the same output directory to merge results seamlessly.
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


def run_experiment_009_part2():
    """Run Experiment 009 Part 2: Experiments 22-42."""
    
    start_time = datetime.now()
    
    print("=" * 80)
    print("EXPERIMENT 009 PART 2: Experiments 22-42 (Continuation)")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Use the EXISTING output directory from Part 1
    output_dir = Path(__file__).parent / "data" / "exp_009_20251019_232730"
    if not output_dir.exists():
        print(f"❌ ERROR: Output directory not found: {output_dir}")
        print("Please update the directory name in this script!")
        return
    
    print(f"[OUTPUT DIR] {output_dir} (continuing from Part 1)")
    print()
    
    # Load dataset
    print("[LOADING DATASET]")
    workers_df, tasks_df = load_data('didi', max_workers=15000, max_tasks=20000)
    print(f"   ✅ Loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print()
    
    # Fixed global settings
    FIXED_SETTINGS = {
        'normalize_scores': True,
        'enable_diagnostics': False,
        'gamma': 0.5,
        'k': 15
    }
    
    # Define experiments 22-42 ONLY
    experiments = []
    experiment_id = 22
    
    # ========================================================================
    # GROUP E: BALANCED GRID SWEEP (9 experiments)
    # ========================================================================
    print("[GROUP E] Balanced Grid Sweep (Experiments 22-30)")
    
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
    
    print(f"   ✅ 9 experiments configured")
    print()
    
    # ========================================================================
    # GROUP F: HIGH-FAIRNESS EDGE (4 experiments)
    # ========================================================================
    print("[GROUP F] High-Fairness Edge (Experiments 31-34)")
    
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
    
    print(f"   ✅ 4 experiments configured")
    print()
    
    # ========================================================================
    # GROUP G: LOW-FAIRNESS EDGE (4 experiments)
    # ========================================================================
    print("[GROUP G] Low-Fairness Edge (Experiments 35-38)")
    
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
    
    print(f"   ✅ 4 experiments configured")
    print()
    
    # ========================================================================
    # GROUP H: LOW-UTILITY EDGE (4 experiments)
    # ========================================================================
    print("[GROUP H] Low-Utility Edge (Experiments 39-42)")
    
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
    
    print(f"   ✅ 4 experiments configured")
    print()
    
    print("=" * 80)
    print(f"[PART 2 EXPERIMENTS] {len(experiments)} (IDs 22-42)")
    print("=" * 80)
    print()
    
    # Estimate duration
    avg_time_per_exp = 27  # minutes (based on Part 1 average)
    estimated_total_hours = (len(experiments) * avg_time_per_exp) / 60
    print(f"[ESTIMATED DURATION] {estimated_total_hours:.1f} hours")
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
        print(f"[EXPERIMENT {exp['id']}/42] {exp['name']} (Part 2: {i}/{len(experiments)})")
        print("=" * 80)
        print(f"[ID] {exp['id']}")
        print(f"[GROUP] {exp['group']}")
        print(f"[STRATEGY] {exp['strategy']}")
        print(f"[DESCRIPTION] {exp['description']}")
        
        params = exp['config_params']
        print(f"[PARAMETERS]")
        print(f"   L1 (Fairness):     {params.get('fairness_weight')}")
        print(f"   L2 (Starvation):   {params.get('starvation_weight')}")
        print(f"   L3 (Utility):      {params.get('utility_weight')}")
        print(f"   Soft Threshold:    {params.get('soft_threshold')}")
        print(f"   Normalize Scores:  {params.get('normalize_scores')}")
        print(f"   Gamma (EWMA):      {params.get('gamma')}")
        print()
        
        try:
            # Create configuration
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
                'fairness_weight': exp['config_params'].get('fairness_weight'),
                'starvation_weight': exp['config_params'].get('starvation_weight'),
                'utility_weight': exp['config_params'].get('utility_weight'),
                'soft_threshold': exp['config_params'].get('soft_threshold'),
                'normalize_scores': exp['config_params'].get('normalize_scores'),
                'gamma': exp['config_params'].get('gamma'),
                
                # Primary metrics
                'completed_tasks': summary.get('completed_tasks', 0),
                'task_assignment_ratio': summary.get('completed_tasks', 0) / 20000,
                'jains_fairness_index': summary.get('jfi', 0),
                'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
                'mean_pickup_distance_km': summary.get('avg_pickup_distance_km', 0),
                'total_travel_km': summary.get('total_travel_distance_km', 0),
                'peak_backlog': summary.get('backlog_peak', 0),
                
                # Metadata
                'duration_seconds': exp_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            result['mean_worker_idle_time_min'] = None
            
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
            
            # Save worker-level data
            if 'workers_df' in summary:
                worker_filename = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_workers.csv"
                summary['workers_df'].to_csv(worker_filename, index=False)
            
            # Progress update
            elapsed_hours = (time.time() - start_time.timestamp()) / 3600
            remaining_experiments = len(experiments) - i
            avg_time_so_far = elapsed_hours / i * 60
            estimated_remaining_hours = (remaining_experiments * avg_time_so_far) / 60
            
            print(f"[PROGRESS] Part 2: {i}/{len(experiments)} complete | Overall: {20+successful_count}/42")
            print(f"[TIME] Part 2 elapsed: {elapsed_hours:.1f}h | Estimated remaining: {estimated_remaining_hours:.1f}h")
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
    # UPDATE AGGREGATE RESULTS
    # ========================================================================
    print("=" * 80)
    print("[UPDATING AGGREGATE RESULTS]")
    print("=" * 80)
    
    # Load existing aggregate CSV
    aggregate_file = output_dir / "experiment_009_aggregate_results.csv"
    if aggregate_file.exists():
        existing_df = pd.read_csv(aggregate_file)
        print(f"   Loaded existing results: {len(existing_df)} experiments")
    else:
        existing_df = pd.DataFrame()
        print(f"   No existing aggregate file found")
    
    # Create new results DataFrame
    new_df = pd.DataFrame(results)
    
    # Calculate mean idle times from worker CSVs
    for idx, row in new_df.iterrows():
        exp_id = row['experiment_id']
        name = row['name']
        worker_file = output_dir / f'exp_{exp_id:03d}_{name}_workers.csv'
        if worker_file.exists():
            workers_df_result = pd.read_csv(worker_file)
            new_df.at[idx, 'mean_worker_idle_time_min'] = workers_df_result['T_idle'].mean()
    
    # Combine with existing results
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values('experiment_id')
    
    # Save updated aggregate CSV
    combined_df.to_csv(aggregate_file, index=False)
    print(f"✅ Updated aggregate results: {aggregate_file}")
    print(f"   Total experiments: {len(combined_df)} (Part 1: 21, Part 2: {len(new_df)})")
    
    # Update experiment manifest
    manifest_file = output_dir / "experiment_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {}
    
    manifest.update({
        'part_2_start_time': start_time.isoformat(),
        'part_2_end_time': datetime.now().isoformat(),
        'part_2_duration_hours': (time.time() - start_time.timestamp()) / 3600,
        'part_2_experiments': len(experiments),
        'part_2_successful': successful_count,
        'part_2_failed': len(failed_experiments),
        'total_experiments_completed': len(combined_df),
    })
    
    if failed_experiments:
        manifest['part_2_failures'] = failed_experiments
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"✅ Updated experiment manifest: {manifest_file}")
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600
    
    print("=" * 80)
    print("[EXPERIMENT 009 PART 2 COMPLETE]")
    print("=" * 80)
    print(f"[START TIME]  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[END TIME]    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION]    {total_duration:.2f} hours")
    print()
    print(f"[PART 2]      {successful_count}/{len(experiments)} experiments successful")
    print(f"[OVERALL]     {len(combined_df)}/42 total experiments completed")
    if failed_experiments:
        print(f"[FAILURES]    {len(failed_experiments)} experiments failed:")
        for failure in failed_experiments:
            print(f"              - Exp {failure['id']} ({failure['name']}): {failure['error']}")
    print()
    print(f"[OUTPUT DIR]  {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_009_part2()



