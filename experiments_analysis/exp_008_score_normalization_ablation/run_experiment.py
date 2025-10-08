#!/usr/bin/env python3
"""
Experiment 008: Score Normalization and Threshold Ablation
==========================================================

Diagnostic experiment to resolve the worker idle time paradox observed in Experiment 006.

This experiment tests two primary hypotheses:
1. Mis-scaled composite score components cause fairness to dominate utility
2. Soft-threshold deferral mechanism creates artificial task shortages

Experimental Groups:
- Group A: Greedy baseline (efficiency reference)
- Group B: Composite current (replicate paradox)
- Group C: Composite + normalization (test H1)
- Group D: Composite + normalization + no threshold (test H1+H2)

Expected Duration: 3-4 hours (12 experiments total)
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


def run_experiment_008():
    """Run Experiment 008: Score Normalization and Threshold Ablation."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("EXPERIMENT 008: Score Normalization and Threshold Ablation")
    print("=" * 80)
    print(f"[START TIME] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[OBJECTIVE] Diagnose and resolve worker idle time paradox")
    print("[FOCUS] Test hypotheses about mis-scaled scores and soft-threshold delays")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / f"exp_008_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT DIR] {output_dir}")
    print()
    
    # Load dataset once for all experiments
    print("[LOADING DATASET]")
    print("   Dataset: DiDi (15,000 workers, 20,000 tasks)")
    workers_df, tasks_df = load_data('didi', max_workers=15000, max_tasks=20000)
    print(f"   ✅ Loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print()
    
    # Define experimental configurations
    # Sweet spot parameters from Experiment 006
    sweet_spot_params = {
        'fairness_weight': 0.5,
        'starvation_weight': 0.8,
        'utility_weight': 0.8,
        'soft_threshold': 0.5,
        'gamma': 0.5,
        'k': 15
    }
    
    experiments = []
    experiment_id = 1
    
    # Group A: Greedy Baseline (3 replications)
    print("[GROUP A] Greedy Baseline")
    print("   Purpose: Efficiency reference")
    for run in range(1, 4):
        experiments.append({
            'id': experiment_id,
            'group': 'A',
            'name': f'Greedy_Run_{run}',
            'description': f'Greedy baseline, Run {run}/3',
            'strategy': 'greedy',
            'config_params': {},
            'normalize_scores': None,
            'disable_soft_threshold': None,
            'run': run
        })
        experiment_id += 1
    print(f"   ✅ {len([e for e in experiments if e['group'] == 'A'])} experiments configured")
    print()
    
    # Group B: Composite Current (replicate paradox) (3 replications)
    print("[GROUP B] Composite Current (Paradox Replication)")
    print("   Purpose: Confirm paradox reproducibility")
    print("   normalize_scores=False, disable_soft_threshold=False")
    for run in range(1, 4):
        experiments.append({
            'id': experiment_id,
            'group': 'B',
            'name': f'Composite_Current_Run_{run}',
            'description': f'Composite current (paradox), Run {run}/3',
            'strategy': 'composite',
            'config_params': {
                **sweet_spot_params,
                'normalize_scores': False,
                'disable_soft_threshold': False
            },
            'normalize_scores': False,
            'disable_soft_threshold': False,
            'run': run
        })
        experiment_id += 1
    print(f"   ✅ {len([e for e in experiments if e['group'] == 'B'])} experiments configured")
    print()
    
    # Group C: Composite + Normalization (test H1) (3 replications)
    print("[GROUP C] Composite + Normalization (Test Hypothesis 1)")
    print("   Purpose: Test if score normalization resolves the paradox")
    print("   normalize_scores=True, disable_soft_threshold=False")
    for run in range(1, 4):
        experiments.append({
            'id': experiment_id,
            'group': 'C',
            'name': f'Composite_Normalized_Run_{run}',
            'description': f'Composite with normalization, Run {run}/3',
            'strategy': 'composite',
            'config_params': {
                **sweet_spot_params,
                'normalize_scores': True,
                'disable_soft_threshold': False
            },
            'normalize_scores': True,
            'disable_soft_threshold': False,
            'run': run
        })
        experiment_id += 1
    print(f"   ✅ {len([e for e in experiments if e['group'] == 'C'])} experiments configured")
    print()
    
    # Group D: Composite + Normalization + No Threshold (test H1+H2) (3 replications)
    print("[GROUP D] Composite + Normalization + No Threshold (Test Both Hypotheses)")
    print("   Purpose: Test if combined intervention fully resolves the paradox")
    print("   normalize_scores=True, disable_soft_threshold=True")
    for run in range(1, 4):
        experiments.append({
            'id': experiment_id,
            'group': 'D',
            'name': f'Composite_Normalized_NoThreshold_Run_{run}',
            'description': f'Composite with normalization and no threshold, Run {run}/3',
            'strategy': 'composite',
            'config_params': {
                **sweet_spot_params,
                'normalize_scores': True,
                'disable_soft_threshold': True
            },
            'normalize_scores': True,
            'disable_soft_threshold': True,
            'run': run
        })
        experiment_id += 1
    print(f"   ✅ {len([e for e in experiments if e['group'] == 'D'])} experiments configured")
    print()
    
    print(f"[TOTAL EXPERIMENTS] {len(experiments)}")
    print()
    
    # Run experiments
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, exp in enumerate(experiments, 1):
        print("=" * 80)
        print(f"[EXPERIMENT {exp['id']}/{len(experiments)}] {exp['name']}")
        print(f"   Group: {exp['group']} | {exp['description']}")
        print("=" * 80)
        
        exp_start_time = time.time()
        
        try:
            # Create config for this experiment
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
                'run': exp['run'],
                
                # Experimental parameters
                'normalize_scores': exp.get('normalize_scores'),
                'disable_soft_threshold': exp.get('disable_soft_threshold'),
                'fairness_weight': exp['config_params'].get('fairness_weight'),
                'starvation_weight': exp['config_params'].get('starvation_weight'),
                'utility_weight': exp['config_params'].get('utility_weight'),
                'soft_threshold': exp['config_params'].get('soft_threshold'),
                'gamma': exp['config_params'].get('gamma'),
                
                # Primary validation metrics
                'mean_worker_idle_time_min': None,  # Will be calculated from worker data
                'completed_tasks': summary.get('completed_tasks', 0),
                'task_assignment_ratio': summary.get('completed_tasks', 0) / 20000,
                
                # Secondary metrics
                'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                'mean_task_wait_time_min': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
                'mean_pickup_distance_km': summary.get('empty_km', 0) / max(1, summary.get('completed_tasks', 1)),
                'total_travel_km': summary.get('total_travel_km', 0),
                
                # System metrics
                'peak_backlog': summary.get('backlog_peak', 0),
                
                # Metadata
                'duration_seconds': exp_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            # Diagnostic metrics (only for Composite strategy)
            if exp['strategy'] == 'composite' and 'diagnostic_summary' in summary:
                diag = summary['diagnostic_summary']
                result.update({
                    'total_assignments': diag.get('total_assignments', 0),
                    'total_deferrals': diag.get('total_deferrals', 0),
                    'deferral_rate': diag.get('deferral_rate', 0),
                    'dominant_fairness_pct': diag.get('dominance_percentages', {}).get('fairness', 0),
                    'dominant_starvation_pct': diag.get('dominance_percentages', {}).get('starvation', 0),
                    'dominant_utility_pct': diag.get('dominance_percentages', {}).get('utility', 0),
                    'avg_dominance_ratio': diag.get('overall_avg_dominance_ratio', 0),
                    'mean_final_score': diag.get('score_statistics', {}).get('mean_final_score', 0),
                })
                
                # Component statistics
                comp_stats = diag.get('component_statistics_raw', {})
                for component in ['fairness', 'starvation', 'utility']:
                    if component in comp_stats:
                        result[f'{component}_raw_mean'] = comp_stats[component].get('mean', 0)
                        result[f'{component}_raw_std'] = comp_stats[component].get('std', 0)
                        result[f'{component}_raw_min'] = comp_stats[component].get('min', 0)
                        result[f'{component}_raw_max'] = comp_stats[component].get('max', 0)
            
            # Calculate worker idle time from worker data
            # (This will be done in analysis notebook from full worker DataFrame)
            
            results.append(result)
            successful_count += 1
            
            # Save individual experiment results
            exp_filename = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_summary.json"
            with open(exp_filename, 'w') as f:
                # Include full summary for detailed analysis
                full_result = {
                    **result,
                    'full_summary': {k: v for k, v in summary.items() if k not in ['metric_tracker', 'diagnostic_tracker']}
                }
                json.dump(full_result, f, indent=2, default=str)
            
            # Save diagnostic data if available
            if exp['strategy'] == 'composite' and 'diagnostic_tracker' in summary:
                tracker = summary['diagnostic_tracker']
                
                # Save assignment records
                assignments_df = tracker.to_dataframe('assignments')
                if not assignments_df.empty:
                    assignments_file = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_assignments.csv"
                    assignments_df.to_csv(assignments_file, index=False)
                    print(f"   ✅ Saved {len(assignments_df):,} assignment records")
                
                # Save deferral records
                deferrals_df = tracker.to_dataframe('deferrals')
                if not deferrals_df.empty:
                    deferrals_file = output_dir / f"exp_{exp['id']:03d}_{exp['name']}_deferrals.csv"
                    deferrals_df.to_csv(deferrals_file, index=False)
                    print(f"   ✅ Saved {len(deferrals_df):,} deferral records")
            
            print(f"[SAVED] {exp_filename}")
            print()
            
        except Exception as e:
            print(f"[ERROR] Experiment {exp['id']} failed:")
            print(f"   {str(e)}")
            traceback.print_exc()
            failed_count += 1
            
            # Record failure
            results.append({
                'experiment_id': exp['id'],
                'group': exp['group'],
                'name': exp['name'],
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            print()
    
    # Save aggregate results
    print("=" * 80)
    print("[SAVING AGGREGATE RESULTS]")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    aggregate_file = output_dir / "experiment_008_aggregate_results.csv"
    results_df.to_csv(aggregate_file, index=False)
    print(f"   ✅ Saved aggregate results: {aggregate_file}")
    
    # Save metadata
    metadata = {
        'experiment_name': 'Experiment 008: Score Normalization and Threshold Ablation',
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration_hours': (datetime.now() - start_time).total_seconds() / 3600,
        'total_experiments': len(experiments),
        'successful': successful_count,
        'failed': failed_count,
        'dataset': 'DiDi (15,000 workers, 20,000 tasks)',
        'output_directory': str(output_dir),
        'sweet_spot_params': sweet_spot_params,
        'groups': {
            'A': 'Greedy baseline (efficiency reference)',
            'B': 'Composite current (replicate paradox)',
            'C': 'Composite + normalization (test H1)',
            'D': 'Composite + normalization + no threshold (test H1+H2)'
        }
    }
    
    metadata_file = output_dir / "experiment_008_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"   ✅ Saved metadata: {metadata_file}")
    print()
    
    # Print summary
    print("=" * 80)
    print("[EXPERIMENT 008 COMPLETE]")
    print("=" * 80)
    print(f"   Total Duration: {metadata['duration_hours']:.2f} hours")
    print(f"   Successful: {successful_count}/{len(experiments)}")
    print(f"   Failed: {failed_count}/{len(experiments)}")
    print(f"   Output Directory: {output_dir}")
    print()
    print("[NEXT STEPS]")
    print("   1. Run analysis.ipynb to visualize results")
    print("   2. Compare idle times across groups A, B, C, D")
    print("   3. Analyze diagnostic metrics (dominance, deferrals)")
    print("   4. Determine which hypothesis is supported")
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_008()

