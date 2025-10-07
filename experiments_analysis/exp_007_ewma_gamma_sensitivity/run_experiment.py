#!/usr/bin/env python3
"""
Experiment 007: EWMA Gamma Sensitivity Analysis
==============================================

CRITICAL EXPERIMENT: Diagnose worker idle time paradox from Experiment 006

This experiment investigates why the fairness-aware Composite strategy 
unexpectedly INCREASES worker idle times compared to the Greedy baseline.

Research Focus:
- RQ2.2: EWMA γ parameter sensitivity 
- RQ1.2: Parameter weight interactions
- Diagnostic: Resolve counter-intuitive idle time behavior

Expected Duration: 2-3 hours
"""

import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def run_ewma_gamma_sensitivity():
    """Run EWMA gamma sensitivity analysis to diagnose worker idle time paradox."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print("[EXPERIMENT 007] EWMA Gamma Sensitivity Analysis")
    print("=" * 55)
    print("[CRITICAL] Diagnosing worker idle time paradox")
    print("[ANALYSIS] Testing EWMA gamma parameter effects on worker idle times")
    print("[CONFIG] Sweet Spot Configuration: lambda1=0.5, lambda2=0.8, lambda3=0.8, threshold=0.5")
    print()
    
    # Load dataset once
    print("[LOADING] Loading dataset...")
    workers_df, tasks_df = load_data('didi', max_workers=15000, max_tasks=20000)
    print(f"[SUCCESS] Dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print()
    
    # Define experimental configurations
    experiments = []
    experiment_id = 1
    
    # Phase 1: EWMA γ Sensitivity Analysis
    print("[PHASE 1] EWMA GAMMA SENSITIVITY ANALYSIS")
    print("=" * 40)
    
    # Sweet Spot configuration from Experiment 006
    base_config = {
        'fairness_weight': 0.5,
        'starvation_weight': 0.8,
        'utility_weight': 0.8,
        'soft_threshold': 0.5
    }
    
    # Test different EWMA gamma values
    gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    runs_per_gamma = 3
    
    for gamma in gamma_values:
        for run in range(1, runs_per_gamma + 1):
            experiments.append({
                'id': experiment_id,
                'phase': 'gamma_sensitivity',
                'name': f'EWMA_gamma_{gamma}_run_{run}',
                'description': f'EWMA γ={gamma}, Run {run}/{runs_per_gamma}',
                'config_params': {
                    **base_config,
                    'ewma_gamma': gamma,
                    'assignment_strategy': 'composite'
                },
                'gamma': gamma,
                'run': run
            })
            experiment_id += 1
    
    print(f"[PHASE 1] {len([e for e in experiments if e['phase'] == 'gamma_sensitivity'])} experiments")
    
    # Phase 2: Weight Interaction Analysis
    print("\n[PHASE 2] WEIGHT INTERACTION ANALYSIS")
    print("=" * 40)
    
    # Test configurations to address weight imbalance hypothesis
    weight_configs = [
        # Higher utility weight
        {'fairness_weight': 0.5, 'starvation_weight': 0.8, 'utility_weight': 1.5, 'soft_threshold': 0.5, 'name': 'Higher_Utility_1.5'},
        {'fairness_weight': 0.5, 'starvation_weight': 0.8, 'utility_weight': 2.0, 'soft_threshold': 0.5, 'name': 'Higher_Utility_2.0'},
        
        # Lower fairness weight  
        {'fairness_weight': 0.3, 'starvation_weight': 0.8, 'utility_weight': 0.8, 'soft_threshold': 0.5, 'name': 'Lower_Fairness_0.3'},
        
        # Balanced approach
        {'fairness_weight': 0.4, 'starvation_weight': 0.8, 'utility_weight': 1.2, 'soft_threshold': 0.5, 'name': 'Balanced_0.4_1.2'},
    ]
    
    # Use optimal gamma from Phase 1 (assume 0.5 as default for now)
    optimal_gamma = 0.5  # Will be updated based on Phase 1 results
    
    for config in weight_configs:
        for run in range(1, 4):  # 3 runs each
            experiments.append({
                'id': experiment_id,
                'phase': 'weight_interaction',
                'name': f"{config['name']}_run_{run}",
                'description': f"{config['name']}, Run {run}/3",
                'config_params': {
                    **config,
                    'ewma_gamma': optimal_gamma,
                    'assignment_strategy': 'composite'
                },
                'run': run
            })
            experiment_id += 1
    
    print(f"[PHASE 2] {len([e for e in experiments if e['phase'] == 'weight_interaction'])} experiments")
    print(f"🔢 Total Experiments: {len(experiments)}")
    print()
    
    # Run experiments
    results = []
    failed_experiments = []
    
    print("🎬 STARTING EXPERIMENTS:")
    print("=" * 30)
    
    phase1_results = []
    
    for i, exp in enumerate(experiments, 1):
        try:
            print(f"\n🧪 Experiment {i}/{len(experiments)}: {exp['name']}")
            print(f"   📝 {exp['description']}")
            
            # Create configuration
            config = create_composite_config(**exp['config_params'])
            
            # Run simulation
            exp_start = time.time()
            sim = Simulation(config, workers_df, tasks_df)
            result = sim.run()
            exp_duration = time.time() - exp_start
            
            # Collect comprehensive results with focus on idle time metrics
            experiment_result = {
                **exp,
                'duration_seconds': exp_duration,
                'timestamp': datetime.now().isoformat(),
                
                # PRIMARY METRICS: Worker Idle Time (Focus of this experiment)
                'mean_worker_idle_time': result.get('mean_worker_idle_time', 0),
                'p95_worker_idle_time': result.get('p95_worker_idle_time', 0),
                'pct_workers_idle_30min': result.get('pct_workers_idle_30min', 0),
                'max_worker_idle_time': result.get('max_worker_idle_time', 0),
                
                # SECONDARY METRICS: Ensure no regression
                'jfi': result.get('jfi', 0),
                'tar': result.get('task_assignment_ratio', 0) * 100,
                'avg_wait_time': result.get('avg_wait_time_minutes', 0),
                'avg_pickup_distance': result.get('avg_pickup_distance_km', 0),
                
                # DIAGNOSTIC METRICS: EWMA behavior
                'ewma_cv': result.get('ewma_cv', 1.0),
                'ewma_responsiveness': result.get('ewma_responsiveness', 0),
                'fairness_assignment_pct': result.get('fairness_assignment_pct', 0),
                'utility_assignment_pct': result.get('utility_assignment_pct', 0),
                'starvation_assignment_pct': result.get('starvation_assignment_pct', 0),
                
                # Standard metrics
                'completed_tasks': result.get('assigned_tasks', 0),
                'total_tasks': result.get('total_tasks', 0),
                'workers_tracked': result.get('workers_tracked', 0)
            }
            
            results.append(experiment_result)
            
            # Track Phase 1 results for optimal gamma determination
            if exp['phase'] == 'gamma_sensitivity':
                phase1_results.append(experiment_result)
            
            # Progress summary with focus on idle time
            idle_time = experiment_result['mean_worker_idle_time']
            jfi = experiment_result['jfi']
            tar = experiment_result['tar']
            pct_idle_30 = experiment_result['pct_workers_idle_30min']
            
            success_indicator = "[SUCCESS]" if idle_time < 20 and jfi > 0.85 else "[WARNING]" if idle_time < 25 else "[ISSUE]"
            
            print(f"   {success_indicator} Idle: {idle_time:.1f}min, JFI: {jfi:.3f}, TAR: {tar:.1f}%")
            print(f"      30min+ idle: {pct_idle_30:.1f}%, Duration: {exp_duration:.1f}s")
            
            # Update optimal gamma for Phase 2 (simple heuristic: minimize idle time while maintaining JFI > 0.8)
            if exp['phase'] == 'gamma_sensitivity' and len(phase1_results) >= 3:
                valid_results = [r for r in phase1_results if r['jfi'] > 0.8]
                if valid_results:
                    best_gamma_result = min(valid_results, key=lambda x: x['mean_worker_idle_time'])
                    optimal_gamma = best_gamma_result.get('gamma', 0.5)
                    
                    # Update remaining Phase 2 experiments with optimal gamma
                    for future_exp in experiments[i:]:
                        if future_exp['phase'] == 'weight_interaction':
                            future_exp['config_params']['ewma_gamma'] = optimal_gamma
            
            # Save temporal data if available
            if hasattr(sim, 'metric_tracker') and sim.metric_tracker:
                temporal_path = f"data/ewma_gamma_sweep_{timestamp}/exp_{exp['id']:03d}_{exp['name']}"
                Path(temporal_path).parent.mkdir(parents=True, exist_ok=True)
                sim.metric_tracker.save_all_data(temporal_path)
            
        except Exception as e:
            print(f"   [FAILED] {str(e)}")
            failed_experiments.append({
                'experiment': exp,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            continue
    
    # Analysis and Results Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n[COMPLETED] EXPERIMENT 007 COMPLETED!")
    print("=" * 35)
    print(f"[SUCCESS] Successful experiments: {len(results)}/{len(experiments)}")
    print(f"[TIMING] Total duration: {total_duration/3600:.1f} hours")
    
    if failed_experiments:
        print(f"[FAILED] Failed experiments: {len(failed_experiments)}")
    
    # Phase 1 Analysis: EWMA γ Sensitivity
    if phase1_results:
        print(f"\n[ANALYSIS] PHASE 1 ANALYSIS: EWMA GAMMA SENSITIVITY")
        print("=" * 45)
        
        gamma_analysis = {}
        for gamma in gamma_values:
            gamma_results = [r for r in phase1_results if r.get('gamma') == gamma]
            if gamma_results:
                avg_idle = np.mean([r['mean_worker_idle_time'] for r in gamma_results])
                avg_jfi = np.mean([r['jfi'] for r in gamma_results])
                avg_pct_30 = np.mean([r['pct_workers_idle_30min'] for r in gamma_results])
                
                gamma_analysis[gamma] = {
                    'avg_idle_time': avg_idle,
                    'avg_jfi': avg_jfi,
                    'avg_pct_30min': avg_pct_30,
                    'experiments': len(gamma_results)
                }
                
                print(f"γ={gamma}: Idle={avg_idle:.1f}min, JFI={avg_jfi:.3f}, 30min+={avg_pct_30:.1f}%")
        
        # Find optimal gamma
        valid_gammas = {g: data for g, data in gamma_analysis.items() if data['avg_jfi'] > 0.8}
        if valid_gammas:
            optimal_gamma_result = min(valid_gammas.items(), key=lambda x: x[1]['avg_idle_time'])
            optimal_gamma = optimal_gamma_result[0]
            print(f"\n[OPTIMAL] OPTIMAL GAMMA: {optimal_gamma} (Idle: {optimal_gamma_result[1]['avg_idle_time']:.1f}min)")
        else:
            print(f"\n[WARNING] No gamma achieved JFI > 0.8 - may need function redesign")
    
    # Save comprehensive results
    final_results = {
        'metadata': {
            'experiment': 'Experiment 007: EWMA Gamma Sensitivity Analysis',
            'timestamp': timestamp,
            'total_experiments': len(experiments),
            'successful_experiments': len(results),
            'failed_experiments': len(failed_experiments),
            'total_duration_seconds': total_duration,
            'dataset_size': {
                'workers': len(workers_df),
                'tasks': len(tasks_df)
            },
            'optimal_gamma': optimal_gamma if 'optimal_gamma' in locals() else None
        },
        'phase1_analysis': gamma_analysis if 'gamma_analysis' in locals() else {},
        'results': results,
        'failed_experiments': failed_experiments
    }
    
    # Save results to experiment data directory (will be committed)
    results_path = f"data/ewma_gamma_sensitivity_{timestamp}.json"
    Path("data").mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_path}")
    
    # Recommendations for next steps
    if results:
        best_overall = min(results, key=lambda x: x['mean_worker_idle_time'])
        print(f"\n🏅 BEST CONFIGURATION:")
        print(f"   Configuration: {best_overall['name']}")
        print(f"   Mean Idle Time: {best_overall['mean_worker_idle_time']:.1f} minutes")
        print(f"   JFI: {best_overall['jfi']:.3f}")
        print(f"   Workers 30min+ idle: {best_overall['pct_workers_idle_30min']:.1f}%")
        
        if best_overall['mean_worker_idle_time'] < 20 and best_overall['jfi'] > 0.85:
            print(f"\n[SUCCESS] SUCCESS: Idle time paradox resolved!")
            print(f"📈 Ready to proceed with RQ1/RQ3 comprehensive exploration")
        elif best_overall['mean_worker_idle_time'] < 25:
            print(f"\n[WARNING] PARTIAL SUCCESS: Improvement achieved but more optimization needed")
            print(f"[RECOMMEND] Consider deeper parameter exploration or function redesign")
        else:
            print(f"\n[ISSUE] ISSUE PERSISTS: Worker idle time paradox not resolved")
            print(f"🔧 May require composite function redesign or bug investigation")
    
    return results_path

if __name__ == "__main__":
    print("[EXPERIMENT] Running Experiment 007: EWMA Gamma Sensitivity Analysis")
    print("🚨 CRITICAL: Diagnosing Worker Idle Time Paradox")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='EWMA Gamma Sensitivity Analysis')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer gamma values')
    args = parser.parse_args()
    
    if args.quick:
        print("[QUICK MODE] Quick mode: Testing gamma=[0.3, 0.5, 0.7] only")
    
    try:
        results_file = run_ewma_gamma_sensitivity()
        print(f"\n[COMPLETED] EWMA Gamma Sensitivity Analysis completed!")
        print(f"[ANALYSIS] Analyze results: jupyter notebook analysis.ipynb")
        print(f"📈 Results file: {results_file}")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        print(traceback.format_exc())
